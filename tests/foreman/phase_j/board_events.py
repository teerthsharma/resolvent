import json
from datetime import datetime

def board_events():
    """Generate board events for each row."""
    events = [
        {
            "t": "test",
            "agent": "Chase",
            "status": "green",
            "name": "FETCH-CHASE-P1"
        },
        {
            "t": "test",
            "agent": "Chase",
            "status": "green",
            "name": "FETCH-CHASE-P2"
        },
        {
            "t": "test",
            "agent": "Chase",
            "status": "red",
            "name": "FETCH-CHASE-P3"
        }
    ]
    return events

if __name__ == '__main__':
    events = board_events()
    for event in events:
        print(json.dumps(event))
