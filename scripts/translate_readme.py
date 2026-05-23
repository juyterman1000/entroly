import os
import time
from openai import OpenAI

def translate_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Create mapping of languages
    languages = {
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'nl': 'Dutch',
        'fr': 'French'
    }

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    for code, lang in languages.items():
        print(f"Translating to {lang}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a technical translator. Translate the following GitHub README into {lang}. Do not translate names like 'Entroly', 'Cursor', 'Claude', 'OpenClaw'. Preserve all markdown formatting, tables, HTML tags, and code blocks exactly as they are without modifying the syntax."},
                    {"role": "user", "content": content}
                ],
                temperature=0.2
            )
            translated = response.choices[0].message.content
            
            # Write to file
            filename = f"README_{code}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(translated)
            print(f"✅ Saved {filename}")
            
        except Exception as e:
            print(f"Error translating {lang}: {e}")

if __name__ == "__main__":
    translate_readme()
