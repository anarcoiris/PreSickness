import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath('services/unified_app'))

from processor import process_hybrid_full
from db import batch_store_raw_messages
from events import parse_whatsapp_line

async def test():
    parsed_messages = []
    current_msg = None
    
    file_path = r'c:\Users\soyko\Documents\PreSickness\datos\paciente2_whatsapp.txt'
    with open(file_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    print('Parsing...')
    for line in lines:
        parsed = parse_whatsapp_line(line)
        if parsed:
            if current_msg:
                parsed_messages.append(current_msg)
            current_msg = {
                'date': parsed['date'], 
                'content': parsed['content'], 
                'metadata': {'sender': parsed['sender']}
            }
        elif current_msg:
            current_msg['content'] += '\n' + line
    
    if current_msg:
        parsed_messages.append(current_msg)

    print('Storing raw messages:', len(parsed_messages))
    
    # Needs a DB pool connection!
    import db
    await db.get_pool()
    
    await batch_store_raw_messages('paciente2', parsed_messages)
    
    print('Processing NLP Pipeline via process_hybrid_full...')
    result = await process_hybrid_full('paciente2')
    print('Result:', result)
    print('Done!')

asyncio.run(test())
