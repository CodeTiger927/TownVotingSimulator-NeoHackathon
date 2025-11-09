#!/bin/bash


set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=========================================="
echo "Town Hall Conversation Runner"
echo "=========================================="
echo ""

if ! curl -s "${BASE_URL}/" > /dev/null 2>&1; then
    echo "Error: Server is not running at ${BASE_URL}"
    echo "Please start the server with: python main.py"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <topic> [politician_1_message] [politician_2_message]"
    echo ""
    echo "Example:"
    echo "  $0 immigration \"We need open borders\" \"We need strict controls\""
    echo ""
    echo "Using default topic: immigration"
    TOPIC="immigration"
else
    TOPIC="$1"
fi

if [ -z "$2" ]; then
    P1_MESSAGE="I believe in welcoming diversity and open borders with proper background checks. Immigration strengthens our community and brings new perspectives and skills."
else
    P1_MESSAGE="$2"
fi

if [ -z "$3" ]; then
    P2_MESSAGE="I support strict immigration controls and border security. We must protect our village's safety and ensure only vetted individuals can enter."
else
    P2_MESSAGE="$3"
fi

echo "Topic: ${TOPIC}"
echo ""
echo "Politician 1 Message:"
echo "  ${P1_MESSAGE}"
echo ""
echo "Politician 2 Message:"
echo "  ${P2_MESSAGE}"
echo ""
echo "Starting town hall conversation..."
echo "=========================================="
echo ""

RESPONSE=$(curl -s -X POST "${BASE_URL}/townhall" \
    -H "Content-Type: application/json" \
    -d "{
        \"topic\": \"${TOPIC}\",
        \"politician_1_message\": \"${P1_MESSAGE}\",
        \"politician_2_message\": \"${P2_MESSAGE}\"
    }")

if [ $? -ne 0 ]; then
    echo "Error: Failed to run town hall"
    exit 1
fi

echo "${RESPONSE}" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)
    
    print('TOWN HALL RESULTS')
    print('=' * 60)
    print(f\"Topic: {data.get('topic', 'N/A')}\")
    print(f\"Turn Order: {', '.join(data.get('turn_order', []))}\")
    print()
    
    print('AGENT RESPONSES:')
    print('-' * 60)
    for response in data.get('agent_responses', []):
        agent = response.get('agent', 'Unknown')
        message = response.get('response', 'No response')
        
        import re
        message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL).strip()
        
        persuasion_deltas = response.get('persuasion_deltas', {})
        
        print(f\"\n{agent}:\")
        if message:
            print(f\"  {message[:300]}{'...' if len(message) > 300 else ''}\")
        else:
            print(f\"  [No response or response was only thinking]\")
    
    print()
    print('=' * 60)
    print('Town hall conversation complete!')
    print()
    print('Next steps:')
    print('  - Check agent summaries: curl ${BASE_URL}/debug/agent/<agent_name>')
    print('  - Run automatic voting: curl -X POST ${BASE_URL}/vote/auto')
    print('  - View results: curl ${BASE_URL}/results')
    
except json.JSONDecodeError as e:
    print(f'Error parsing response: {e}')
    print('Raw response:')
    print(sys.stdin.read())
    sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error displaying results. Raw response:"
    echo "${RESPONSE}"
    exit 1
fi
