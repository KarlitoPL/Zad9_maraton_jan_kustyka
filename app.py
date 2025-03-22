import streamlit as st
import pandas as pd
import base64
import json

from dotenv import dotenv_values
from openai import OpenAI
import instructor
from pydantic import BaseModel

from langfuse import Langfuse
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Brakuje klucza OPENAI_API_KEY w środowisku!")

llm_client = OpenAI(api_key=api_key)


def configure_clients():
    return langfuse  # już nie potrzebujesz instructor_client
# ---------------------------------------
# MODELE I FUNKCJE POMOCNICZE
# ---------------------------------------

class ParseInputResponse(BaseModel):
    Imię: str
    Nazwisko: str
    Miasto: str
    Drużyna: str
    Wiek: int
    Płeć: str
    Czas: int  # czas w sekundach
    Dystans: float  # dystans w kilometrach



def parse_input_with_gpt(user_text: str, langfuse_client) -> dict:
    """Wywołuje GPT i śledzi z Langfuse cały proces parsowania."""
    system_prompt = (
       "Jesteś asystentem, który otrzymuje swobodnie napisany tekst użytkownika i ma za zadanie wyodrębnić dane:\n"
        "\n"
        "1) Wyciągnij dystans pokonany przez użytkownika jako liczbę w kilometrach (float).\n"
        "   - Akceptowane formaty: „10 km”, „5 kilometrów”, „5000 metrów” itp.\n"
        "   - Zamień wszystko na kilometry (np. 5000 metrów → 5.0). Jeśli nie podano, ustaw 0.0.\n"
        "\n"
        "2) Wyciągnij czas jako liczbę sekund (int).\n"
        "   - Akceptowane formaty: „45 minut”, „1h 30min”, „01:35:00”, „3600s” itp.\n"
        "   - Zamień wszystko na liczbę sekund (np. 45 minut → 2700). Jeśli nie podano, ustaw 0.\n"
        "\n"
        "Nie obliczaj czasu półmaratonu (`Czas_sec`) – to zrobi kod.\n"
        "\n"
        "3) Dodatkowe dane:\n"
        "   - Wiek: jeśli brak → 0,\n"
        "   - Płeć: M lub K. Domyślnie przyjmij M, ale jeśli imię lub tekst wskazuje wyraźnie na kobietę (np. Janina), wybierz K. Spróbuj rozpoznać płeć na podstawie imienia i kontekstu, nawet jeśli nie została jawnie podana.\n"
        "   - Imię: jeśli brak → \"Anonimowe\",\n"
        "   - Nazwisko: jeśli brak → \"Anonimowe\",\n"
        "   - Miasto: jeśli brak → \"Nie podano\",\n"
        "   - Drużyna: jeśli brak → \"Brak drużyny\".\n"
        "\n"
        "Zwróć WYŁĄCZNIE obiekt JSON w formacie:\n"
        "{\"Wiek\":..., \"Płeć\":\"...\", \"Imię\":\"...\", \"Nazwisko\":\"...\", \"Miasto\":\"...\", \"Drużyna\":\"...\", \"Czas\":..., \"Dystans\":...}\n"
        "\n"
        "Bez komentarzy, bez dodatkowych pól, bez tłumaczenia. Tylko czysty JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    # Langfuse trace
    trace = langfuse_client.trace(name="parse_user_input", input=messages)
    span = trace.span(name="gpt-call", input={"user_text": user_text})

    try:
        completion = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )

        response = completion.choices[0].message.content

        span.end(
            output=response,
            usage={
                "input": completion.usage.prompt_tokens,
                "output": completion.usage.completion_tokens,
                "total": completion.usage.total_tokens,
                "unit": "TOKENS"
            }
        )

        trace.update(output=response)

        return json.loads(response)
    except Exception as e:
        span.end(output={"error": str(e)})
        trace.update(output={"error": str(e)})
        return {"error": f"Langfuse GPT error: {str(e)}"}

def seconds_to_hhmmss(seconds: float) -> str:
    """Konwersja sekund do formatu hh:mm:ss (z zerami wiodącymi)."""
    total_sec = int(round(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def load_data() -> pd.DataFrame:
    """Wczytuje dane z CSV, przetwarza kolumny i zwraca DataFrame."""
    df = pd.read_csv("df_2024_final.csv")
    df["Miejsce"] = df["Miejsce"].astype(int)
    df["km/h"] = df["km/h"].round(2)
    return df

def process_ranking(final_time_sec, final_gender, final_city, final_team, final_name1, final_name2, final_age, df_rank) -> dict:
    user_category = f"{final_gender}{str(final_age)[0]}0"
    best_time_sec = df_rank["Czas_sec"].min()

    # Obliczenie tempa
    user_speed = round(21.0975 / (final_time_sec / 3600), 2)
    best_speed = round(21.0975 / (best_time_sec / 3600), 2)
    time_diff = best_time_sec - final_time_sec

    # Dodajemy nowy wiersz użytkownika
    user_row = {
        "Imię": final_name1,
        "Nazwisko": final_name2,
        "Miasto": final_city,
        "Płeć": final_gender,
        "Wiek": final_age,
        "Drużyna": final_team,
        "Kategoria wiekowa": user_category,
        "Czas_sec": final_time_sec,
        "Czas": seconds_to_hhmmss(final_time_sec),
        "km/h": user_speed
    }

    df_rank_with_user = pd.concat([df_rank, pd.DataFrame([user_row])], ignore_index=True)

    # Sortowanie po czasie
    df_sorted = df_rank_with_user.sort_values("Czas_sec").reset_index(drop=True)
    df_sorted["Miejsce"] = df_sorted.index + 1

    # Znalezienie indeksu użytkownika
    user_index = df_sorted[
        (df_sorted["Imię"] == final_name1) &
        (df_sorted["Nazwisko"] == final_name2) &
        (df_sorted["Czas_sec"] == final_time_sec)
    ].index[0]
    user_place = int(df_sorted.loc[user_index, "Miejsce"])

    return {
        "df_rank_sorted": df_sorted,
        "user_place": user_place,
        "user_index": user_index,
        "user_speed": user_speed,
        "best_speed": best_speed,
        "best_time_sec": best_time_sec,
        "time_diff": time_diff,
    }


def highlight_user(row, user_place):
    """Funkcja stylizująca wiersz użytkownika w tabeli."""
    if row["Miejsce"] == user_place:
        return ['background-color: rgba(255, 215, 0, 0.3)'] * len(row)
    return [''] * len(row)

def highlight_user(row, user_place):
    if row["Miejsce"] == user_place:
        return ['background-color: rgba(255, 215, 0, 0.3)'] * len(row)
    return [''] * len(row)

def display_ranking(df_rank_sorted, user_index, user_place):
    st.markdown("<h3 style='text-align: left;'>🏆 TOP RANKING</h3>", unsafe_allow_html=True)

    ranking_cols = [
        "Miejsce", "Czas", "km/h", "Imię", "Nazwisko",
        "Miasto", "Płeć", "Wiek", "Drużyna", "Kategoria wiekowa"
    ]

    # Top tabela
    if user_place <= 10:
        top_n = df_rank_sorted.head(10)[ranking_cols]
    else:
        top_n = df_rank_sorted.head(5)[ranking_cols]

    st.dataframe(
        top_n.style.hide(axis="index")
            .format({"km/h": "{:.2f}"})
            .apply(lambda row: highlight_user(row, user_place), axis=1)
    )

    if user_place > 10:
        st.markdown("<h4 style='text-align: left;'>📊 Twoja pozycja w rankingu</h4>", unsafe_allow_html=True)
        lower = max(user_index - 5, 0)
        upper = min(user_index + 5, len(df_rank_sorted) - 1)
        nearby = df_rank_sorted.iloc[lower:upper + 1][ranking_cols]

        st.dataframe(
            nearby.style.hide(axis="index")
                .format({"km/h": "{:.2f}"})
                .apply(lambda row: highlight_user(row, user_place), axis=1)
        )







# ---------------------------------------
# GŁÓWNA FUNKCJA APLIKACJI
# ---------------------------------------

def main():
    langfuse_client = configure_clients()
    df_rank = load_data()

    # --- NAGŁÓWEK ---
    st.markdown(
        """
        <h2 style="color:#333; text-align:center; font-weight:300;">
            <span style="color:#28a745; font-weight:500;">Jan Kuśtyka</span> spojrzy na Twój czas...
            <br>... i powie Ci, co czeka Cię na kolejnym <strong style="color:#FF5733;">półmaratonie we Wrocławiu</strong>.
        </h2>
        """,
        unsafe_allow_html=True
    )

    # --- OPIS ---
    st.markdown(
        """
        <p style="font-size:1rem; color:#555; text-align:center; line-height:1.6;">
            Opowiedz o sobie – jak przy zapisach na bieg.<br>
            <strong>Imię. Nazwisko. Miasto. Drużyna. Wiek. Płeć. Ostatni wynik i na jakim dystansie.</strong><br>
            W dowolnym formacie. Pisz swobodnie – my to ogarniemy.
        </p>
        """,
        unsafe_allow_html=True
    )

    # --- INPUT ---
    st.markdown("<h4>📝 <strong>WPISZ SWOJE DANE:</strong></h4>", unsafe_allow_html=True)
    user_input = st.text_input("Wpisz swoje dane", label_visibility="collapsed")    



    # --- PRZYCISK ---
    if st.button("Oblicz ranking"):
        user_text = user_input.strip()
        if not user_text:
            st.warning("Nic nie wpisałeś!")
        else:
            gpt_result = parse_input_with_gpt(user_text, langfuse_client)


            czas_w_sekundach = float(gpt_result.get("Czas", 0))
            dystans_w_km = float(gpt_result.get("Dystans", 0))

            if czas_w_sekundach > 0 and dystans_w_km > 0:
                tempo_sec_per_km = czas_w_sekundach / dystans_w_km
                czas_na_polmaraton = tempo_sec_per_km * 21.0975
                gpt_result["Czas_sec"] = int(czas_na_polmaraton)
            else:
                gpt_result["Czas_sec"] = 0

            final_time_sec = gpt_result["Czas_sec"]

            # with st.expander("🔍 Co GPT wyciągnął z Twoich danych?"):
                # st.json(gpt_result)
            
            if gpt_result["Czas_sec"] < 1800:
                st.warning("⚠️ Szybko biegasz, ale niestety Super-Bohaterowie jak Ty nie mogą startować w tym maratonie. ")

            final_age = gpt_result["Wiek"]
            final_gender = gpt_result["Płeć"]
            final_time_sec = gpt_result["Czas_sec"]
            final_city = gpt_result.get("Miasto", "Nie podano")
            final_team = gpt_result.get("Drużyna", "Brak drużyny")
            final_name1 = gpt_result.get("Imię", "Anonimowe")
            final_name2 = gpt_result.get("Nazwisko", "Anonimowe")

            # UPPERCASE
            final_name1 = final_name1.upper()
            final_name2 = final_name2.upper()
            final_city = final_city.upper()

            if final_time_sec <= 0:
                st.info("Przeciążenie serwerów, spróbuj jeszcze raz.")
            else:
                result = process_ranking(
                    final_time_sec, final_gender, final_city, final_team,
                    final_name1, final_name2, final_age, df_rank
                )
                if "error" in result:
                    st.error(result["error"])
                else:
                    user_place = result["user_place"]
                    user_index = result["user_index"]
                    user_speed = result["user_speed"]
                    best_speed = result["best_speed"]
                    best_time_sec = result["best_time_sec"]
                    time_diff = result["time_diff"]
                    df_rank_sorted = result["df_rank_sorted"]

                    # --- SEKCJA WYNIKU ---
                    st.markdown("<hr style='margin-top:30px; margin-bottom:20px;'>", unsafe_allow_html=True)
                    st.markdown(f"### Twój czas w pół-maratonie to **{seconds_to_hhmmss(final_time_sec)}**")
                    st.markdown(f"## 🏅 **Miejsce: {user_place}**")

                    if user_place == 1:
                        st.markdown("## 🎉 **GRATULUJĘ ZWYCIĘSTWA!** 🎉")
                        st.markdown(f"### Twoje tempo: **{user_speed} km/h**")
                    else:
                        formatted_time_diff = seconds_to_hhmmss(abs(time_diff))
                        st.markdown(f"#### Aby wygrać, musisz pobiec **o {formatted_time_diff} szybciej**.")
                        st.markdown(
                            f"📈 **Wystarczy, że zwiększysz tempo z {user_speed} km/h do {round(best_speed + 0.01, 2)} km/h!**"
                        )

                    display_ranking(df_rank_sorted, user_index, user_place)

    

    # --- MAPA ---
    st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Trasa półmaratonu we Wrocławiu", unsafe_allow_html=True)

    # WSTAW GRAFIKĘ MAPY (jeśli masz mapę jako plik PNG / JPG)
    try:
        st.image("mapa.png", use_container_width=True)
    except:
        st.info("Mapa trasy nie została jeszcze dodana.")

if __name__ == "__main__":
    main()