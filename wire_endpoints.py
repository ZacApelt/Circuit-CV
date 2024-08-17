import cv2
import numpy as np
from collections import deque

# Load the binary image
image = cv2.imread('./circuits/tut1_blank.png', 0)

# Invert the image if needed (now you have white lines on a black background)
binary_image = cv2.bitwise_not(image)

thinned = cv2.ximgproc.thinning(binary_image)

# for every white pixel in the thinned image, check if it is an endpoint by checking how many neighbours it has
# if it has only one neighbour, it is an endpoint

# put a red circle on image at centre


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
                id += 1
            
print(endpoints)


def find_connected_endpoints(skeleton, endpoints):
    visited = np.zeros_like(skeleton, dtype=bool)
    connections = []

    def bfs(start_id, start_pos):
        print("BFS")
        queue = deque([start_pos])
        visited[start_pos[1], start_pos[0]] = True
        while queue:
            x, y = queue.popleft()

            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        print("nope")
                        continue

                    nx, ny = x + dx, y + dy
                    print(nx,ny)

                    if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                        if skeleton[ny, nx] == 255 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

                            # Check if the current point is an endpoint
                            for endpoint in endpoints:
                                if endpoint['pos'] == (nx, ny) and endpoint['id'] != start_id:
                                    connections.append([start_id, endpoint['id']])
                                    break
        # Iterate through each endpoint and perform BFS to find connectionsfor endpoint in endpoints:
        if not visited[endpoint['pos'][1], endpoint['pos'][0]]:
            bfs(endpoint['id'], endpoint['pos'])

    return connections

connections = find_connected_endpoints(thinned, endpoints)
print(connections)

cv2.imshow('Endpoints', thinned)
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()