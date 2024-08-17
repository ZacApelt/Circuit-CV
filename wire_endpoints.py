import cv2
import numpy as np
from collections import deque

# Load the binary image
image = cv2.imread('./circuits/tut1_blank.png', 0)

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

sample_OCR = [{'component': 'R', 'value': 100, 'corners': [[120, 140], [150, 140], [150, 160], [120, 160]]}, {'component': 'C', 'value': 0.1, 'corners': [[300, 50], [330, 50], [330, 70], [300, 70]]}]
# write the OCR result on the image
for i in range(len(sample_OCR)):
    cv2.rectangle(image, sample_OCR[i]['corners'][0], sample_OCR[i]['corners'][2], (0,255,0), 2)
    cv2.putText(image, sample_OCR[i]['component'] + str(sample_OCR[i]['value']), (sample_OCR[i]['corners'][0][0], sample_OCR[i]['corners'][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

sample_components = [{"component": "R", "corners": [(278, 20), (331,50)]}, {"component": "R", "corners": [(80, 144), (110,210)]}]
# draw a rectangle around the component
for component in sample_components:
    cv2.rectangle(image, component['corners'][0], component['corners'][1], (0,0,255), 2)

# match component to nodes -> adds a new key to the component dictionary corresponding to endpoint id
for i in range(len(sample_components)):
    # find mid point of all sides of component
    mid_points = [0, 0, 0, 0]
    corners = sample_components[i]['corners']
    mid_points[0] = ((corners[0][0] + corners[1][0])//2, corners[0][1])
    mid_points[1] = (corners[1][0], (corners[0][1] + corners[1][1])//2)
    mid_points[2] = ((corners[0][0] + corners[1][0])//2, corners[1][1])
    mid_points[3] = (corners[0][0], (corners[0][1] + corners[1][1])//2)

    # find distances between mid points and endpoints
    distances = []
    for connection in all_connections:
        for endpoint_id in connection:
            endpoint = endpoints[endpoint_id]
            for mid_point in mid_points:
                distance = np.sqrt((mid_point[0] - endpoint['pos'][0])**2 + (mid_point[1] - endpoint['pos'][1])**2)
                distances.append((endpoint_id, distance))
    # find the 2 closest endpoints
    distances.sort(key=lambda x: x[1])
    # print the id of the 2 closest endpoints
    sample_components[i]['connections'] = [distances[0][0], distances[1][0]]

    # match the OCR result to each component
    # find the centroid of the component
    ocr_distance = []
    centroid_component = [(sample_components[i]['corners'][0][0] + sample_components[i]['corners'][1][0])//2, (sample_components[i]['corners'][0][1] + sample_components[i]['corners'][1][1])//2]
    # draw a circle at the centroid
    cv2.circle(image, (centroid_component[0], centroid_component[1]), 5, (0,0,255))
               
    for j in range(len(sample_OCR)):
        # find the centroid of the OCR result
        centroid_OCR = [(sample_OCR[j]['corners'][0][0] + sample_OCR[j]['corners'][1][0])//2, (sample_OCR[j]['corners'][0][1] + sample_OCR[j]['corners'][1][1])//2]
        # draw a circle at the centroid
        cv2.circle(image, (centroid_OCR[0], centroid_OCR[1]), 5, (0,0,255))

        # find the distance between the centroids
        distance = np.sqrt((centroid_component[0] - centroid_OCR[0])**2 + (centroid_component[1] - centroid_OCR[1])**2)
        
        if distance < 100:
            ocr_distance.append((j, distance))
    
    # find the closest OCR result
    ocr_distance.sort(key=lambda x: x[1])
    if ocr_distance:
        sample_components[i]['OCR classification'] = sample_OCR[ocr_distance[0][0]]['component']
        sample_components[i]['value'] = sample_OCR[ocr_distance[0][0]]['value']
        


print(sample_components)


cv2.imshow('Endpoints', thinned)
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()