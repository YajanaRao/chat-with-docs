from script import query_rag

def test_ticket_to_ride_rules():
  query = "What are the projects ?"
  response = query_rag(query, "Serenity, Harmony, and Peace") 
  print(response)
