import glob
import os

pages = glob.glob('pages/*.py')

replacements = {
    'ticket_priority': 'priority',
    'resolution_time': 'resolution_time_hours',
    'ticket_description': 'issue_description',
    'ticket_subject': 'category',
    'subject_length': 'category_length'
}

for page in pages:
    with open(page, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(page, 'w', encoding='utf-8') as f:
        f.write(content)

print("Streamlit pages updated successfully!")
