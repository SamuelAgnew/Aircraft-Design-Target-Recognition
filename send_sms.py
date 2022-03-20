from twilio.rest import Client
client = Client("AC~~~~~~~~~~~~~~~~~", "0d~~~~~~~~~~~~~~~~~~")
client.messages.create(to="~~~~~~~~~~~~~", 
                       from_="~~~~~~~~~~~", 
                       body="Hello ____! I have the results from your Image Recognition Test!")
                       
