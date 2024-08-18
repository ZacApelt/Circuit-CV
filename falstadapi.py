import urllib.parse

def create_falstad_url(file_path):
    """
    Create a Falstad Circuit Simulator URL with a circuit encoded from a file.

    Args:
        file_path (str): Path to the text file containing the circuit data.

    Returns:
        str: URL to load the circuit in Falstad Circuit Simulator.
    """
    # Read the circuit data from the file
    with open(file_path, 'r') as file:
        circuit_data = file.read()
    
    # URL-encode the circuit data
    encoded_data = urllib.parse.quote(circuit_data)
    
    # Construct the Falstad URL
    falstad_url = f"https://www.falstad.com/circuit/circuitjs.html?cct={encoded_data}"
    
    return falstad_url

# Example usage
file_path = 'demo.txt'
url = create_falstad_url(file_path)
print(url)
