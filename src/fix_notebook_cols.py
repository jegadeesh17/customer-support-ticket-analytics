import json
import glob
import os

notebooks = glob.glob('notebooks/*.ipynb')

replacements = {
    'ticket_priority': 'priority',
    'resolution_time': 'resolution_time_hours',
    'ticket_description': 'issue_description',
    'ticket_subject': 'category',
    'subject_length': 'category_length'
}

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace synthesized block to prevent it from synthesizing random target if column is found
    # Actually, the replacement of `resolution_time` with `resolution_time_hours` covers it.
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(nb_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Notebooks updated successfully!")
