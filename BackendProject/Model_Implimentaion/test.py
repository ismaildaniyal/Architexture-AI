import os
import spacy
from dotenv import load_dotenv
import google.generativeai as genai
import json
import numpy as np
import torch
import pandas as pd
# Load spaCy's English model
nlp = spacy.load("en_core_web_lg")

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def normalize_adjacency(adj):
    
    input_is_numpy = isinstance(adj, np.ndarray)

    # Convert NumPy array to PyTorch tensor if necessary
    if input_is_numpy:
        adj = torch.tensor(adj, dtype=torch.float32)

    # Add self-loops
    adj = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree matrix
    degree = torch.sum(adj, dim=1)  # Degree for each node
    degree_inv_sqrt = torch.pow(degree, -0.5)  # D^(-1/2)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0  # Handle division by zero

    # Create D^(-1/2) * A * D^(-1/2)
    d_mat_inv_sqrt = torch.diag(degree_inv_sqrt)
    normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    # Convert back to NumPy if the input was a NumPy array
    if input_is_numpy:
        return normalized_adj.cpu().numpy()
    return normalized_adj
# Feature extraction functions
def extract_room_type(doc):
    for token in doc:
        if token.text.lower().startswith(("bedroom", "livingroom", "balcony", "kitchen", "washroom", "studyroom", "closet", "storage", "corridor")):
            return token.text.capitalize()
    return "Unknown"

def extract_direction(doc):
    for token in doc:
        if token.text.lower() in {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "center"}:
            return token.text.capitalize()
    return "Unknown"


def extract_dimension(doc):
    for token in doc:
        if token.text.lower() == "size":
            next_token = token.nbor(1)  # Look two tokens ahead to get the size value
            if next_token.like_num:  # Check if the token is a number
                return next_token.text  # Return the number as the size
    return "Unknown"



def extract_adjacent_rooms(doc):
    adjacent_rooms = []
    capture = False

    for token in doc:
        if token.text.lower() == "to":
            capture = True
            continue

        if capture:
            if token.text == ".":
                break
            if token.pos_ == "PROPN" or (token.dep_ in {"appos", "conj"} and token.pos_ == "PROPN"):
                adjacent_rooms.append(token.text)
    return adjacent_rooms if adjacent_rooms else ["Unknown"]

T_real = ['Center', 'Northeast', 'South', 'West', 'Northwest', 'North', 'East', 'Southwest', 'Southeast']
c = ['livingroom', 'kitchen', 'balcony', 'bedroom', 'washroom', 'studyroom', 'closet', 'storage', 'corridor']

pos_type_encoded = pd.get_dummies(T_real).astype(int)
room_type_encoded = pd.get_dummies(c).astype(int)

def clean_room_type(room_type):
    cleaned_room_type = ''
    for char in room_type:
        if not char.isdigit():
            cleaned_room_type += char.lower()  # Add non-digit characters to the cleaned string
    return cleaned_room_type

# Function to validate and enhance house plan input

def validate_and_enhance_house_plan(user_input):
    # Configure the Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.01,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 5000,
            "response_mime_type": "text/plain",
        },
    )
    # Start a chat session
    chat_session = model.start_chat(history=[])
    print("user_input",user_input)
    # Validation step
    validation_response = chat_session.send_message(
        f"""
        You are a helpful assistant for generating and validating house plans. 
        Input: {user_input} 
        Please give me 1 if the count and types of rooms and do not ask for multi-floor. 
        If the input is valid, respond only with "1". Do not write any kind of code.
        If invalid, provide "0" and a detailed reason for invalidation (e.g.room count mismatch, or multi-floor request). 
        """
    )
    print("validation_response",validation_response)
    # Process validation response
    validation_text = validation_response.text.strip()
    if validation_text.startswith("0"):
        print(f"Input is invalid: {validation_text[1:].strip()}")
        reason = validation_text[1:].strip()  # Extract the reason
        return {"is_valid":  False, "reason": reason}  # Return the reason if invalid

    # Enhancement step
    enhancement_response = chat_session.send_message(
        f"""
        By viewing input Assign a sequence of room names (e.g., roomname1, roomname2, etc.) to the same type of room. For example, if there any room with count , give it Roomname1 same for others. 
        Assign size in sq meters in range minimum 6 and maximum 30 also include decimal based. Assign the best directions with no collision
        (North, South, East, West, Northeast, Northwest, Southeast, Southwest, and center) to each room.Do Not Add Hallway. 
        Change word  bathroom to washroom if in input.Do not add any room other than mentioned by user.Change Central to center.Change study room to studyroom.
        Add other adjacencies rather than the mentioned ones each room must have adjacency with some room even if you have to add by your ownself that is logical.
        Provide the details in one single paragraph only.
        Input: {user_input}
        """
    )
    print("Enhancement_response", enhancement_response)
    # Reformat the enhanced paragraph
    formatted_response = chat_session.send_message(
        f"""
        Now from previous paragraph 
        Change in proper format 
        -"RoomName1 is in direction with size e.g 48." Mention attachments as: "It is attached to RoomName."  
        RoomName2 detail on separate Line  
        Do not add bullets
        Previous response: {enhancement_response.text.strip()}
        """
    )
    print("Responsed",formatted_response)
    # Process each line of the formatted response for feature extraction
    formatted_lines = formatted_response.text.strip().split("\n")
    # print(f"Total number of lines: {len(formatted_lines)}")
    # print("Final Output:")
    room_matrix = []
    adjacency_matrix = np.zeros((len(formatted_lines), len(formatted_lines)), dtype=int)

    # Create a mapping from room names to their indices
    # room_name_to_id = {line.split()[0].lower(): idx for idx, line in enumerate(formatted_lines)}
    room_name_to_id = {}

# Loop through the formatted_lines and extract room namess (the first word of each line)
    for idx, line in enumerate(formatted_lines):
        room_name = line.split()[0].lower()  # Extract the first word as the room name
        room_name_to_id[room_name] = idx

    for i, line in enumerate(formatted_lines):
        doc = nlp(line)
        room_type = extract_room_type(doc)
        direction = extract_direction(doc)
        dimension = extract_dimension(doc)
        adjacent_rooms = extract_adjacent_rooms(doc)
        cleaned_room_type = clean_room_type(room_type)
        room_type_vector = np.array(room_type_encoded[cleaned_room_type]).reshape(-1)
        position_vector = np.array(pos_type_encoded[direction]).reshape(-1)
        # Append room details to the room matrix
        room_matrix.append(np.concatenate([room_type_vector, np.array([float(dimension)]), position_vector]))

        # For each adjacent room, mark the adjacency in the adjacency matrix
        for adj_room in adjacent_rooms:
            adj_id = room_name_to_id.get(adj_room.lower(), None)
            # adjacency_matrix[i][i] = 1
            if adj_id is not None:
                adjacency_matrix[i][adj_id] = 1
                adjacency_matrix[adj_id][i] = 1  # Ensure symmetry
        # print(room_matrix)

    print("before norm adjacency_matrix",adjacency_matrix)
    adjacency_matrix=normalize_adjacency(adjacency_matrix)
    for i in range(len(adjacency_matrix)):
        adjacency_matrix[i][i] = 1

    print("adjacency_matrix",adjacency_matrix)
    room_vec=[]
    padded_matrix = np.zeros((14, len(room_type_encoded.columns) + len(pos_type_encoded.columns) + 1), dtype=object)  # Room type + position + size
    for idx, row in enumerate(room_matrix):
        padded_matrix[idx] = row
    adjvec=[]
    adj_pad_matrix = np.zeros((14, 14), dtype=object)

# Ensure the size of the adjacency matrix is correct
    for idx, row in enumerate(adjacency_matrix):
        adj_pad_matrix[idx, :len(row)] = np.array(row)
    adjvec.append(adj_pad_matrix)
    room_vec.append(padded_matrix)
    adj=np.array(adjvec[0])
    vec=np.array(room_vec[0])
    # print("Room Matrix:" + str(vec))
    # print("Adjacency Matrix:" + str(adj))
    print("adj before load",adj)
    np.save('C:/Users/SMART TECH/Documents/FYP/BackendProject/Model_Implimentaion/adjacency_matrix.npy', adj)
    np.save('C:/Users/SMART TECH/Documents/FYP/BackendProject/Model_Implimentaion/room_matrix.npy', vec)
    print("Ismail")
    return {"is_valid": True, "reason": "No"} 


# Example usage
# if __name__ == "__main__":
#     user_input = (
#                     "My house is 5 marla. I need 2 bedrooms, 1 washroom, 1 kitchen, and a livingroom.")
#     validate_and_enhance_house_plan(user_input)
