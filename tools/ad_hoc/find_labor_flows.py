#!/usr/bin/env python3
"""Find ISTAT dataflow IDs for labor indicators."""

import requests
import xml.etree.ElementTree as ET

# Get ISTAT dataflows
print("Fetching ISTAT dataflows...")
response = requests.get("https://esploradati.istat.it/SDMXWS/rest/dataflow/IT1", timeout=120)
root = ET.fromstring(response.text)

# Define namespaces
ns = {
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "c": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}

# Labor-related keywords
keywords = ["disoccup", "unemployment", "occupaz", "neet", "employment", "lavoro"]

results = []
for df in root.findall(".//s:Dataflow", ns):
    df_id = df.get("id", "")
    struct_ref = df.find(".//s:Structure/*[@id]", ns)
    struct_id = struct_ref.get("id", "") if struct_ref is not None else ""

    it_name = ""
    en_name = ""
    for name in df.findall("c:Name", ns):
        lang = name.get("{http://www.w3.org/XML/1998/namespace}lang", "")
        text = name.text or ""
        if lang == "it":
            it_name = text
        elif lang == "en":
            en_name = text

    search_text = f"{df_id} {struct_id} {it_name} {en_name}".lower()
    if any(k in search_text for k in keywords):
        results.append((df_id, struct_id, it_name[:50], en_name[:50]))

print(f"\nFound {len(results)} labor-related dataflows:\n")
print(f"{'Dataflow ID':<45} | {'Structure ID':<45} | {'Name':<50}")
print("=" * 150)
for df_id, struct, it_name, en_name in sorted(results):
    name = it_name or en_name
    print(f"{df_id:<45} | {struct:<45} | {name:<50}")
#!/usr/bin/env python3
"""Find ISTAT dataflow IDs for labor indicators."""

import requests
import xml.etree.ElementTree as ET

# Get ISTAT dataflows
print("Fetching ISTAT dataflows...")
response = requests.get("https://esploradati.istat.it/SDMXWS/rest/dataflow/IT1", timeout=120)
root = ET.fromstring(response.text)

# Define namespaces
ns = {
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "c": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}

# Labor-related keywords
keywords = ["disoccup", "unemployment", "occupaz", "neet", "employment", "lavoro"]

results = []
for df in root.findall(".//s:Dataflow", ns):
    df_id = df.get("id", "")
    # Get structure ID
    struct_ref = df.find(".//s:Structure/*[@id]", ns)
    struct_id = struct_ref.get("id", "") if struct_ref is not None else ""

    # Get Italian name
    it_name = ""
    for name in df.findall('c:Name[@xml:lang="it"]', ns):
        it_name = name.text or ""
        break

    en_name = ""
    for name in df.findall('c:Name[@xml:lang="en"]', ns):
        en_name = name.text or ""
        break
    # Get Italian and English names
    it_name = ""
    en_name = ""
    for name in df.findall("c:Name", ns):
        lang = name.get("{http://www.w3.org/XML/1998/namespace}lang", "")
        if lang == "it":
            it_name = name.text or ""
        elif lang == "en":
            en_name = name.text or ""
    # Check if it's labor-related
    search_text = f"{df_id} {struct_id} {it_name} {en_name}".lower()
    if any(k in search_text for k in keywords):
        results.append((df_id, struct_id, it_name[:50], en_name[:50]))

# Print results
print(f"\nFound {len(results)} labor-related dataflows:\n")
print(f"{'Dataflow ID':<45} | {'Structure ID':<45} | {'Name':<50}")
print("=" * 150)
for df_id, struct, it_name, en_name in sorted(results):
    name = it_name or en_name
    print(f"{df_id:<45} | {struct:<45} | {name:<50}")
