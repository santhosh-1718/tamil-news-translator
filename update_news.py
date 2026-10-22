import feedparser
import torch
import json
import os
from datetime import datetime, timedelta
from dateutil import parser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Setup
RSS_URLS = [
    "https://tamil.news18.com/commonfeeds/v1/tam/rss/national.xml",
    "https://tamil.oneindia.com/rss/feeds/oneindia-tamil-fb.xml"
]
HOURS_TO_KEEP = 24
cutoff_time = datetime.now() - timedelta(hours=HOURS_TO_KEEP)
new_entries = []

# 2. Fetch News
print("Fetching news...")
for url in RSS_URLS:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        try:
            pub_date = parser.parse(entry.published).replace(tzinfo=None)
            if pub_date > cutoff_time:
                new_entries.append(entry)
        except:
            pass

new_entries.sort(key=lambda x: parser.parse(x.published).replace(tzinfo=None), reverse=True)
new_entries = new_entries[:10] # Limit to 10 for GitHub Action memory limits

# 3. Load Model (CPU mode for GitHub Actions)
print("Loading Model...")
model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.src_lang = "tam_Taml"

translated_news = []

# 4. Translate
print("Translating...")
for entry in new_entries:
    tamil_text = entry.title
    inputs = tokenizer(tamil_text, return_tensors="pt")
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
    
    english_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    translated_news.append({
        "tamil": tamil_text,
        "english": english_text,
        "link": entry.link,
        "published": parser.parse(entry.published).strftime("%d %b %Y, %I:%M %p")
    })

# 5. Generate HTML
print("Generating HTML...")
html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; padding: 20px; }}
  .container {{ max-width: 800px; margin: auto; }}
  .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: white; border-radius: 10px; border-top: 5px solid #16263a; }}
  .card {{ background: white; padding: 20px; margin-bottom: 15px; border-radius: 8px; border-left: 5px solid #d9891b; }}
  .english-title {{ font-size: 1.25rem; font-weight: bold; color: #1a1a1a; margin-bottom: 10px; }}
  .tamil-title {{ font-size: 1rem; color: #666; margin-bottom: 15px; font-style: italic; }}
  .meta {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: #888; }}
  a {{ color: #0066cc; text-decoration: none; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1 style="margin:0 0 10px 0; color:#16263a;">Live Tamil News Translation</h1>
    <p style="margin:0 0 5px 0; color:#555;">Sourced from News18 & OneIndia</p>
    <p style="margin:0; font-size:0.85rem; color:#888;">Updated: {datetime.now().strftime("%d %b %Y, %I:%M %p")}</p>
  </div>
"""

for item in translated_news:
    html_content += f"""
  <div class="card">
    <div class="english-title">{item['english']}</div>
    <div class="tamil-title">{item['tamil']}</div>
    <div class="meta">
      <span>{item['published']}</span>
      <a href="{item['link']}" target="_blank">Read Original &rarr;</a>
    </div>
  </div>"""

html_content += "</div></body></html>"

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Finished!")
