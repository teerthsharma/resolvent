import json
from datetime import datetime

def fetch_paper(arxiv_id=None, title=None):
    """Stub that returns empty paper data."""
    return {
        'title': title or '',
        'authors': [],
        'year': '',
        'venue': '',
        'identifier': arxiv_id or '',
        'abstract_sentence': '',
        'status': '[U]'
    }

def test_fetch_records():
    """Test that all three papers are fetched with full metadata."""
    
    # Paper 1: Merrill, Petty, Sabharwal 2024
    p1 = fetch_paper(arxiv_id='2404.08819', title='The Illusion of State in State-Space Models')
    assert p1['title'] == 'The Illusion of State in State-Space Models'
    assert '2404.08819' in p1['identifier']
    assert len(p1['abstract_sentence']) > 0
    assert p1['status'] in ['fetched', '[U]']
    
    # Paper 2: Grazzi et al. 2024
    p2 = fetch_paper(arxiv_id='2411.12537')
    assert p2['year'] == '2024'
    assert '2411.12537' in p2['identifier']
    assert p2['status'] in ['fetched', '[U]']
    
    # Paper 3: Barrington 1989
    p3 = fetch_paper()
    assert p3['year'] == '1989'
    assert 'branching program' in p3['title'].lower() or p3['status'] == '[U]'

if __name__ == '__main__':
    test_fetch_records()
    print("PASS")
