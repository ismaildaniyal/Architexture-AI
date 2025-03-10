import spacy
import google.generativeai as genai
import numpy as np
import torch
import pandas as pd

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
genai.configure(api_key="AIzaSyD7qoKFwrYX8O7AzE2_3RZxe--gxU8hUoY")  
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
        if token.text.lower().startswith(("bedroom", "livingroom", "balcony", "kitchen", "washroom", "studyroom", "closet", "storage", "corridor",'living' )):
            if(token.text.lower() == 'living'):
                return 'livingroom'
            else:
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
            next_token = token.nbor(1)  # Look for next tokens ahead to get the size value
            if next_token.like_num:
                  # Check if the token is a number
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
            if token.text == "$." or token.text == ".":
                break
            elif token.text == "and":
                continue
            else:
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



    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.01,
            "top_p": 0.95,
            "top_k": 10,
            "max_output_tokens": 5000,
            "response_mime_type": "text/plain",
        },
    )

    # Start a chat session (initialize it only once)
    chat_session = model.start_chat(history=[])

    # user_input = (
    #     "The house consists of four bedrooms, two washrooms, one kitchen, and one living room. Two bedrooms must share one washroom, while the other two have direct access to a private washroom. The living room should be centrally located, connecting all bedrooms. The kitchen should be next to the living room but have a separate entrance from outside. One bedroom should be slightly larger than the others. The house must have at least two external walls with windows" 
    #     )
    check = process_input(user_input)
    print(check)
    # Validation step
    validation_response = chat_session.send_message(
        f"""
        You are a helpful assistant for generating and validating house plans. 
        Input: {user_input} 
        Please give me 1 if the count and types of rooms are given and without mentioning multiple floors.. 
        If the input is valid, respond only with "1".
        If invalid, provide "0" and a detailed reason for invalidation (e.g.room count mismatch, or multi-floor request or ask for code). 
        """
    )

    # Process validation response
    validation_text = validation_response.text.strip()
    if validation_text.startswith("0"):
        print(f"Input is invalid: {validation_text[1:].strip()}")
        reason = validation_text[1:].strip()  # Extract the reason
        return {"is_valid":  False, "reason": reason}  # Return the reason if invalid

    # Enhancement step
    enhancement_response1 = chat_session.send_message(
        f"""
        Input: {user_input}
        => If user has given size direction and adjacency, then assign the room name as per the given input.
        =>By viewing input assign a sequence of room names (e.g., roomname1, roomname2, etc.) to the same type of room. For example, if there any room with count , give it Roomname1 same for others. 
        =>Assign size in sq meters in range minimum 3 and maximum 30 also include decimal based.
        => Washroom  is less than 5 sq ft. Bedroom and kitchen are in range 15 to 20.Living room is 25 to 30 sq ft.
        Do not add any room other than in user input.
        Provide the details in one single paragraph only.
        """
    )
    print(f"Enhancement Response: {enhancement_response1.text.strip()}")
    enhancement_response2 =chat_session.send_message(
        f"""
        Input: {enhancement_response1.text.strip()}
        =>Ensure each room has valid size.
        => First Assign adjacency of each room with others if not given.
        =>Washroom never  attached with kitchen.
        =>Adjacency must be logical and valid.
        => Do not add any room other than in user input.
        =>Generate a minimum adjacency structure for a house layout, ensuring that all rooms remain connected in a single structure. Assign room connections using the Minimum Spanning Tree (MST) approach, minimizing the number of connections while maintaining full accessibility between all rooms.
        Ensure Again it has one single structure.
        Providee details in one single paragraph only.
        """
    )
    enhancement_response3 =chat_session.send_message(
        f"""
        Input: {enhancement_response2.text.strip()}
        =>Ensure each room has  size.(must)
        => Now Based on Adjacency assign direction to each room.
        =>Living room must be in center. It is attached to all rooms.(Only if user ask living room)
        =>Assign directions from predefined types
        (North, South, East, West, Northeast, Northwest, Southeast, Southwest, and center) to each room.
        =>if room is attached to any other one or more room, its direction would be dependent on the attached room..
        => if bedroom is attached to wahroom than direction of washroom is same as bedroom.(Must)
         Do not add any room other than in user input.
        Providee details in one single paragraph only.
        """
    )

    print(f"Updated Input: {enhancement_response2.text.strip()}")
    enhancement_response4 =chat_session.send_message(
        f"""
        Input: {enhancement_response3.text.strip()}
        =>Change living room to livingroom(must). Change study to studyroom. Change bathroom to washroom.Change central to center.(if any)
        Provide details in one single paragraph only.
        Reverify the directions and adjacency.They must be logical and valid.
        Reverify the size of each room.
        Do not add any room other than in user input.
        """
    )
    print(f"Updated Input: {enhancement_response3.text.strip()}")
    # Reformat the enhanced paragraph
    formatted_response = chat_session.send_message(
        f"""
         Do not add any room other than in user input.
        Now from previous paragraph 
        Change in proper format 
        -"(RoomName)1 is in direction with size e.g 48." Mention attachments as: "It is attached to RoomName2, RoomName3, and RoomName4 $."  
        RoomName2 detail on separate Line  
        Follow this format for all rooms.
        Do not add bullets
        Previous response: {enhancement_response4.text.strip()} 
        """
    )
    print(f"Formatted Response: {formatted_response.text.strip()}")
    # Process each line of the formatted response for feature extraction
    formatted_lines = formatted_response.text.strip().split("\n")
    print(f"Total number of lines: {len(formatted_lines)}")
    print("Final Output:")
    room_matrix = []
    adjacency_matrix = np.zeros((len(formatted_lines), len(formatted_lines)), dtype=int)

    # Create a mapping from room names to their indices
    room_name_to_id = {}

    # Loop through the formatted_lines and extract room names (the first word of each line)
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
        print(cleaned_room_type)
        print(dimension)
        # Ensure extracted direction exists in predefined types
        if direction not in T_real:
            print(f"Warning: Invalid direction '{direction}' found. Assigning 'Center'.")
            direction = "Center"

        # Ensure extracted room type exists in predefined categories
        
        if cleaned_room_type not in c:
            print(f"Warning: Invalid room type '{room_type}' found. Assigning 'livingroom'.")
            cleaned_room_type = "livingroom"

# Convert extracted values to vectors safely


        # Get the one-hot encoded vector for the cleaned room type
        room_type_vector = np.array(room_type_encoded[cleaned_room_type]).reshape(-1)
        position_vector = np.array(pos_type_encoded[direction]).reshape(-1)

        # Append room details to the room matrix
        room_matrix.append(np.concatenate([room_type_vector, np.array([float(dimension)]), position_vector]))

        # For each adjacent room, mark the adjacency in the adjacency matrix
        for adj_room in adjacent_rooms:
            adj_id = room_name_to_id.get(adj_room.lower(), None)
            adjacency_matrix[i][i] = 1
            if adj_id is not None:
                adjacency_matrix[i][adj_id] = 1
                adjacency_matrix[adj_id][i] = 1  # Ensure symmetry

        print(f"Room {i+1}: {line}") # Print the original line for context
        print(f"Room Matrix Row {i+1}: {room_matrix[i]}") # Print individual room_matrix rows
        print(f"Adjacency Matrix Row {i+1}: {adjacency_matrix[i]}") # Print individual adjacency_matrix rows

    adjacency_matrix = normalize_adjacency(adjacency_matrix)
    print("adjacency_matrix BY AHMAD:")
    print(adjacency_matrix)
    # Ensure the diagonal entries of the adjacency matrix are set to one
    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i, i] = 1
    room_vec = []
    padded_matrix = np.zeros((14, len(room_type_encoded.columns) + len(pos_type_encoded.columns) + 1), dtype=object)
    for idx, row in enumerate(room_matrix):
        padded_matrix[idx] = row
    adjvec = []
    adj_pad_matrix = np.zeros((14, 14), dtype=object)

    # Ensure the size of the adjacency matrix is correct
    for idx, row in enumerate(adjacency_matrix):
        adj_pad_matrix[idx, :len(row)] = np.array(row)
    adjvec.append(adj_pad_matrix)
    room_vec.append(padded_matrix)
    adj = np.array(adjvec[0])
    vec = np.array(room_vec[0])
    print("Room Matrix:" + str(vec))
    print("Adjacency Matrix:" + str(adj))
    np.save(r'C:\Users\SMART TECH\Desktop\New folder (2)\Architexture-AI\BackendProject\Model_Implimentaion\adjacency_matrix.npy', adj)
    np.save(r'C:\Users\SMART TECH\Desktop\New folder (2)\Architexture-AI\BackendProject\Model_Implimentaion\room_matrix.npy', vec)
    return {"is_valid": True, "reason": "No"} 

def process_input(user_input):
    if "code" in user_input.lower():
        return False
    else:
        return True

# Example usage
# if __name__ == "__main__":
#     # Configure the Gemini model
#     model = genai.GenerativeModel(
#         model_name="gemini-1.5-flash",
#         generation_config={
#             "temperature": 0.01,
#             "top_p": 0.95,
#             "top_k": 10,
#             "max_output_tokens": 5000,
#             "response_mime_type": "text/plain",
#         },
#     )

#     # Start a chat session (initialize it only once)
#     chat_session = model.start_chat(history=[])

#     user_input = (
#         "The house consists of four bedrooms, two washrooms, one kitchen, and one living room. Two bedrooms must share one washroom, while the other two have direct access to a private washroom. The living room should be centrally located, connecting all bedrooms. The kitchen should be next to the living room but have a separate entrance from outside. One bedroom should be slightly larger than the others. The house must have at least two external walls with windows" 
#         )
#     check = process_input(user_input)
#     print(check)

    # if check:
    #     validate_and_enhance_house_plan(user_input, chat_session)

    # # Example of using the same chat session with a follow-up question:
    # follow_up_input = "add one living room in it."  # A follow-up question

    # check_follow_up = process_input(follow_up_input)
    # if check_follow_up:
    #     validate_and_enhance_house_plan(follow_up_input, chat_session)  # Pass the same chat_session