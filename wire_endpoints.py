import cv2
import numpy as np
from collections import deque

# Load the binary image
image = cv2.imread('./circuits/clean_circuit.png', 0)
# Invert the image if needed (now you have white lines on a black background)
binary_image = cv2.bitwise_not(image)
thinned = cv2.ximgproc.thinning(binary_image)


# find the end points
id = 0
endpoints = []
for i in range(1, thinned.shape[0] - 1):
    for j in range(1, thinned.shape[1] - 1):
        if thinned[i, j] == 255:
            # if a white pixel
            neighbours = thinned[i-1:i+2, j-1:j+2]
            #print(neighbours)
            if np.sum(neighbours) == 255*2:
                endpoints.append({"id": id, "pos":(j, i)})
                # put a red pixel at the endpoint
                cv2.circle(image, (j,i), 5, (0,0,255), 2)
                # put id next to the endpoint
                cv2.putText(image, str(id), (j+10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                id += 1
            
print(endpoints)

# connect the end points
def find_connected_endpoints(skeleton, endpoints, start_id):
    visited = np.zeros_like(skeleton, dtype=bool)
    connections = [start_id]
    start_pos = endpoints[start_id]['pos']

    queue = deque([start_pos])
    visited[start_pos[1], start_pos[0]] = True# Mark start position as visited
    while queue:
        x, y = queue.popleft()

        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                    if skeleton[ny, nx] == 255 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

                        # Check if the current point is an endpoint
                        for endpoint in endpoints:
                            if endpoint['pos'] == (nx, ny) and endpoint['id'] != start_id:
                                connections.append(endpoint['id'])
    return connections

# find the max id of the endpoints
max_id = max([endpoint['id'] for endpoint in endpoints])

all_connections = []
all_connections.append(find_connected_endpoints(thinned, endpoints, 0))

# Iterate over each endpoint ID
for i in range(max_id):
    # Check if the endpoint ID is already in any connection list
    if not any(i in connection_list for connection_list in all_connections):
        # Find connected endpoints for the current ID
        connected_endpoints = find_connected_endpoints(thinned, endpoints, i)
        # Append the list of connected endpoints to all_connections
        all_connections.append(connected_endpoints)

print(all_connections)

cv2.imshow('Endpoints', thinned)
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()