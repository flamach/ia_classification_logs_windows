import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Paramètres de génération
N_LOGS = 100_000_000  # 100 millions de logs
sources = [f"App{i}" for i in range(1, 21)]
entry_types = ["Information", "Warning", "Error", "0"]
messages_info = ["Operation completed successfully.", "User login detected.", "File saved."]
messages_warn = ["Low disk space.", "High memory usage.", "Network latency detected."]
messages_error = ["Application crashed.", "Failed to connect to database.", "Permission denied."]
messages_0 = ["Unknown event.", "No details available."]

rows = []
start_time = datetime.now() - timedelta(days=1)
for i in range(N_LOGS):
    entry_type = random.choices(entry_types, weights=[0.78, 0.12, 0.09, 0.01])[0]
    if entry_type == "Information":
        message = random.choice(messages_info)
    elif entry_type == "Warning":
        message = random.choice(messages_warn)
    elif entry_type == "Error":
        message = random.choice(messages_error)
    else:
        message = random.choice(messages_0)
    source = random.choice(sources)
    timestamp = start_time + timedelta(seconds=random.randint(0, 86400))
    rows.append({
        "TimeGenerated": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "EntryType": entry_type,
        "Message": message
    })
    if (i+1) % 500_000 == 0:
        print(f"Générés: {i+1} logs...")

print("Conversion en DataFrame...")
df = pd.DataFrame(rows)
df.to_csv("synthetic_eventlog.csv", index=False)
print("Fichier synthetic_eventlog.csv créé avec succès.")
