from playwright.sync_api import sync_playwright
import pandas as pd
import re
import time

BASE = "https://www.timeanddate.com/weather/india/guwahati/historic"


def extract_number(text):
    match = re.search(r"\d+\.?\d*", text)
    return float(match.group()) if match else None


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto(BASE, wait_until="domcontentloaded")

    # wait for dropdowns
    page.wait_for_selector("#month")
    page.wait_for_selector("#wt-his-select")

    all_data = []

    # ---------------------------
    # STEP 1: GET MONTH OPTIONS
    # ---------------------------
    months = page.query_selector_all("#month option")

    # skip "Past 2 Weeks"
    month_values = [
        m.get_attribute("value")
        for m in months
        if m.get_attribute("value") is not None
    ]

    # limit months (example: first 3 months)
    month_values = month_values[:3]

    for month in month_values:
        print(f"\n📅 Processing month: {month}")

        # ---------------------------
        # SELECT MONTH
        # ---------------------------
        page.select_option("#month", value=month)
        time.sleep(3)

        # ---------------------------
        # GET DAYS
        # ---------------------------
        days = page.query_selector_all("#wt-his-select option")
        day_values = [d.get_attribute("value") for d in days]

        for day in day_values:
            print(f"  → Day: {day}")

            try:
                # ---------------------------
                # SELECT DAY
                # ---------------------------
                page.select_option("#wt-his-select", value=day)

                # wait for table update
                page.wait_for_selector("#wt-his tbody tr")
                time.sleep(2)

                # ---------------------------
                # EXTRACT FIRST ROW
                # ---------------------------
                row = page.query_selector("#wt-his tbody tr")
                cols = row.query_selector_all("td")

                if len(cols) < 7:
                    continue
                weather = cols[2].inner_text().strip()
                temp = extract_number(cols[1].inner_text())
                humidity = extract_number(cols[5].inner_text())
                pressure = extract_number(cols[6].inner_text())

                all_data.append({
                    "date": day,
                    "month": month,
                    "weather":weather,
                    "temp": temp,
                    "humidity": humidity,
                    "pressure": pressure
                })

            except Exception as e:
                print("    ❌ Error:", e)

            time.sleep(1)

    browser.close()


# ---------------------------
# SAVE DATA
# ---------------------------
df = pd.DataFrame(all_data)

print("\nTotal rows:", len(df))
print(df.head())

df.to_csv("weather_multi_months.csv", index=False)

print("✅ DONE")