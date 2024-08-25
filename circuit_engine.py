import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
import easyocr
import numpy as np
from collections import deque
import pprint
import webbrowser
import math
import urllib.parse
from inference_sdk import InferenceHTTPClient
import re


def generate_bboxes(use_api, model, image_path):
    class_dict = {0: 'V', 1: 'arr', 2: 'V',
                  3: 'i', 4: 'L', 5: 'l-', 6: 'R', 7: 'C'}
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        bboxs = cv2.imread(image_path)
    except FileNotFoundError:
        print("Image not found")
        quit()

    if use_api:
        return generate_api_bboxes(class_dict, image, draw, bboxs)
    else:
        return generate_local_bboxes(class_dict, draw, model, image, bboxs)


def preprocessing(image_path):
    # opencv image preprocessing
    image = cv2.imread(image_path)
    # binary filter the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow("binary", binary)

    # Invert the binary image
    binary = cv2.bitwise_not(binary)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours
    for i, contour in enumerate(contours):
        # Get the contour area
        area = cv2.contourArea(contour)

        # Remove small contours, whether outer or inner
        if area < 5:
            cv2.drawContours(binary, [contour], -1, 0, -1)
        # Check if it's an inner contour and remove it if it's below a threshold
        # Adjust this inner contour threshold as needed
        elif hierarchy[0][i][3] != -1 and area < 5:
            cv2.drawContours(binary, [contour], -1, 0, -1)

    # Invert the binary image back to the original state
    binary = cv2.bitwise_not(binary)
    cv2.imwrite("./outputs/prepocessed.png", binary)


def generate_api_bboxes(class_dict, image, draw, bboxs):
    components = []
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="TD5xEPznLTfQhznZz4pf"
    )
    apiresults = CLIENT.infer(
        "./outputs/prepocessed.png", model_id="circuit-recognition/2")
    for detection in apiresults['predictions']:
        double = False
        id = class_dict.get(detection.get("class_id"))
        # Extract the bounding box coordinates
        x = detection['x']
        y = detection['y']
        width = detection['width']
        height = detection['height']

        # Compute the bounding box coordinates
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2

        # Round coordinates
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        for component in components:
            subx = component.get('corners')[0][0]
            suby = component.get('corners')[0][1]
            if math.sqrt(((x2 - subx)**2) + ((y1 - suby)**2)) < 30 and math.sqrt(((x2 - subx)**2) + ((y1 - suby)**2)) != 0:
                double = True
                print(math.sqrt(((x - subx)**2) + ((y - suby)**2)))
        if not double:
            components.append(
                {'component': id, 'corners': [(x2, y1), (x1, y2)]})
    for component in components:
        corner1 = component.get('corners')[0]
        corner2 = component.get('corners')[1]
        draw.rectangle([corner2[0], corner1[1], corner1[0],
                       corner2[1]], fill="white")
        cv2.rectangle(bboxs, (corner2[0], corner1[1]),
                      (corner1[0], corner2[1]), (0, 0, 255), 2)
    image.save("outputs/components_removed.png")
    return components, bboxs


def generate_local_bboxes(class_dict, draw, model, image, bboxs):
    components = []
    results = model.predict(source="./outputs/prepocessed.png")

    # Initialize an empty tensor for bounding boxes
    component_bounding_boxes = torch.empty((0, 4))
    for detection in results[0]:
        id = class_dict[int(detection.boxes.cls.item())]
        
        box = detection.boxes.xyxy  # Extract the bounding box
        component_bounding_boxes = torch.cat(
            (component_bounding_boxes, box), dim=0)

        # Convert to a list of coordinates
        x1, y1, x2, y2 = box[0]
        corners = [
            (round(x2.item()), round(y1.item())),
            (round(x1.item()), round(y2.item()))]

        components.append({'component': id, 'corners': corners})
        org = (round(x2.item()), round(y2.item()))
        cv2.putText(bboxs, id, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    # Create a drawing object

    # Replace each component bounding box with white space
    for box in component_bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")
        cv2.rectangle(bboxs, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
    image.save("outputs/components_removed.png")
    return components, bboxs


def classify_ocr_results(result):
    labels = []

    def split_string(s):
        match = re.match(r"(\d+)([a-zA-Z]+)$", s)
        if match:
            number_part = match.group(1)
            letter_part = match.group(2)
            return number_part, letter_part
        else:
            return '', ''

    def convert_to_float_or_int(value):
        # Use regex to remove non-numeric characters except for the decimal point
        numeric_string = ''.join(re.findall(r'\d+\.?\d*', str(value)))

        if numeric_string:
            # Convert to float if there is a decimal point, otherwise convert to int
            return float(numeric_string) if '.' in numeric_string else int(numeric_string)

    for text1 in result:
        text = text1[1]
        if 'L' in text:
            continue
        if " " in text:
            value, unit = text.split(' ', 2)
        elif not text[0].isnumeric():
            value = int(1)
            unit = text
        elif text[-1].isnumeric():
            value = int(text)
            unit = 'R'
        else:
            value, unit = split_string(text)

        if unit.isnumeric():
            unit = 'R'

        value = convert_to_float_or_int(value)

        if "K" in unit or "k" in unit:
            value *= 1000
        elif "MF" in unit or "mF" in unit or "pF" in unit:  # often mistakes mu for p
            value *= 0.000001
        elif "MH" in unit or "mH" in unit or "pH" in unit:  # often mistakes mu for p
            value *= 0.000001
        elif "M" in unit:
            value *= 1000000
        elif "m" in unit:
            value *= 0.001
        elif "u" in unit:
            value *= 0.000001
        elif "n" in unit:
            value *= 0.000000001
        elif "p" in unit:
            value *= 0.000000000001

        if "V" in unit or "v" in unit:
            labels.append(
                {"component": "V", "value": value, "corners": text1[0]})
        elif "A" in unit:
            labels.append(
                {"component": "A", "value": value, "corners": text1[0]})
        elif "H" in unit:
            labels.append(
                {"component": "L", "value": value, "corners": text1[0]})
        elif "F" in unit:
            labels.append(
                {"component": "C", "value": value, "corners": text1[0]})
        else:
            labels.append(
                {"component": "R", "value": value, "corners": text1[0]})

    return labels


def ocr(bboxs, thresh):
    # Create an OCR reader object
    reader = easyocr.Reader(lang_list=['en'], gpu=False)
    # Load the image with components removed
    component_removed_image = Image.open(
        "outputs/components_removed.png").convert("RGB")
    draw2 = ImageDraw.Draw(component_removed_image)

    # Perform OCR on the image
    ocr_results = reader.readtext(
        image="outputs/components_removed.png", threshold=thresh)

    # Draw bounding boxes and labels on the image
    for result in ocr_results:
        points, text, confidence = result
        points = [tuple(point) for point in points]

        cv2.polylines(bboxs, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2) # draw bounding box
        cv2.putText(bboxs, text, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # draw text

    classified_results = classify_ocr_results(ocr_results)
    for result in classified_results:
        if result.get('value') == None:
            classified_results.remove(result)

    # Initialize an empty tensor for OCR bounding boxes
    ocr_bounding_boxes = torch.empty((0, 4))

    # Extract OCR bounding boxes from the classified results
    for component in classified_results:
        corner_box = component["corners"]

        # Convert corner format to xyxy format
        x_min = min([point[0] for point in corner_box])
        y_min = min([point[1] for point in corner_box])
        x_max = max([point[0] for point in corner_box])
        y_max = max([point[1] for point in corner_box])

        # Create a tensor in xyxy format
        corner_box_tensor = torch.tensor([[x_min, y_min, x_max, y_max]])

        # Concatenate the corner box tensor to the OCR bounding boxes tensor
        ocr_bounding_boxes = torch.cat(
            (ocr_bounding_boxes, corner_box_tensor), dim=0)

    # Replace each OCR bounding box with white space
    for box in ocr_bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        draw2.rectangle([x_min, y_min, x_max, y_max], fill="white")

    # Show the final image with all bounding boxes removed
    # component_removed_image.show()
    component_removed_image.save("./outputs/clean_circuit.png")
    cv2.imwrite("./outputs/bboxs.png", bboxs)
    return classified_results


def calculate_endpoints(classified_results, components):
    endpoint_coordinate_mapping = {}
    all_coordinates = []
    endpoints = []
    all_connections = []
    # Load the binary image
    image = cv2.imread('./outputs/clean_circuit.png', 0)
    # Invert the image if needed (now you have white lines on a black background)
    binary_image = cv2.bitwise_not(image)
    thinned = cv2.ximgproc.thinning(binary_image)
    # cv2.imshow("thinnewd", thinned)

    # find the end points
    id = 0

    for i in range(1, thinned.shape[0] - 1):
        for j in range(1, thinned.shape[1] - 1):
            if thinned[i, j] == 255:
                # if a white pixel
                neighbours = thinned[i-1:i+2, j-1:j+2]
                if np.sum(neighbours) == 255*2:
                    endpoints.append({"id": id, "pos": (j, i)})
                    # put a red pixel at the endpoint
                    cv2.circle(image, (j, i), 5, (0, 0, 255), 2)
                    # put id next to the endpoint
                    cv2.putText(image, str(id), (j+10, i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    id += 1

    # find the max id of the endpoints
    max_id = max([endpoint['id'] for endpoint in endpoints])

    all_connections.append(find_connected_endpoints(thinned, endpoints, 0))

    # Iterate over each endpoint ID
    for i in range(max_id):
        # Check if the endpoint ID is already in any connection list
        if not any(i in connection_list for connection_list in all_connections):
            # Find connected endpoints for the current ID
            connected_endpoints = find_connected_endpoints(
                thinned, endpoints, i)
            # Append the list of connected endpoints to all_connections
            all_connections.append(connected_endpoints)

    # print(all_connections)

    # match component to nodes -> adds a new key to the component dictionary corresponding to endpoint id
    for i in range(len(components)):
        # find mid point of all sides of component
        mid_points = [0, 0, 0, 0]
        corners = components[i]['corners']
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
                    distance = np.sqrt(
                        (mid_point[0] - endpoint['pos'][0])**2 + (mid_point[1] - endpoint['pos'][1])**2)
                    distances.append((endpoint_id, distance))
        # find the 2 closest endpoints
        distances.sort(key=lambda x: x[1])
        # print the id of the 2 closest endpoints
        components[i]['connections'] = [distances[0][0], distances[1][0]]

        # match the OCR result to each component
        # find the centroid of the component
        ocr_distance = []
        centroid_component = [(components[i]['corners'][0][0] + components[i]['corners'][1][0]) //
                              2, (components[i]['corners'][0][1] + components[i]['corners'][1][1])//2]
        # draw a circle at the centroid
        cv2.circle(
            image, (centroid_component[0], centroid_component[1]), 5, (0, 0, 255))

        for j in range(len(classified_results)):
            # find the centroid of the OCR result
            centroid_OCR = [(classified_results[j]['corners'][0][0] + classified_results[j]['corners'][1][0]) //
                            2, (classified_results[j]['corners'][0][1] + classified_results[j]['corners'][1][1])//2]
            # draw a circle at the centroid
            # cv2.circle(image, (centroid_OCR[0], centroid_OCR[1]), 5, (0,0,255))

            # find the distance between the centroids
            distance = np.sqrt(
                (centroid_component[0] - centroid_OCR[0])**2 + (centroid_component[1] - centroid_OCR[1])**2)

            if distance < 200:
                ocr_distance.append((j, distance))

        # find the closest OCR result
        ocr_distance.sort(key=lambda x: x[1])
        if ocr_distance:
            components[i]['OCR classification'] = classified_results[ocr_distance[0][0]]['component']
            components[i]['value'] = classified_results[ocr_distance[0][0]]['value']
        else:
            components[i]['OCR classification'] = 'None'
            components[i]['value'] = 0
    cv2.imwrite("./outputs/endpoints.png", image)

    for group in all_connections:
        group_coordinates = []
        for endpoint in group:
            for point in endpoints:
                if point.get('id') == endpoint:
                    group_coordinates.append(point.get('pos'))
                    endpoint_coordinate_mapping.update(
                        {endpoint: point.get('pos')})
        all_coordinates.append(group_coordinates)

    for component in components:
        x1 = endpoint_coordinate_mapping[component["connections"][0]][0]
        x2 = endpoint_coordinate_mapping[component["connections"][1]][0]
    return endpoints, all_connections, all_coordinates, endpoint_coordinate_mapping


def generating_circuit(all_coordinates, endpoint_coordinate_mapping, components, all_connections):
    falstad_input = '$ 1 0.000005 10.20027730826997 50 5 43 5e-11\n'
    # snapping to a grid
    snapping_factor = 5
    for endpoint in endpoint_coordinate_mapping.keys():
        coords = endpoint_coordinate_mapping.get(endpoint)
        new_point = (math.floor(coords[0]/snapping_factor)*snapping_factor,
                     math.floor(coords[1]/snapping_factor)*snapping_factor)
        endpoint_coordinate_mapping.update({endpoint: new_point})

    # updates all_coordinates to endpoint_coordinate_mapping
    for i, group in enumerate(all_connections):
        for j, point in enumerate(group):
            all_coordinates[i][j] = endpoint_coordinate_mapping[point]

    # Connecting wires between nodes in each group
    for group in all_coordinates:
        group.sort(key=lambda coord: coord[0])
        for i in range(len(group)-1):
            falstad_input += 'w ' + str(group[i][0]) + ' ' + str(group[i][1]) + ' ' + str(
                group[i+1][0]) + ' ' + str(group[i+1][1]) + ' ' + str(0) + '\n'

    falstad_mapping = {'V': 'v', 'arr': 'i', 'C': 'c','i': 'i', 'L': 'l', 'l-': 'l-', 'R': 'r', 'F': 'c'}
    orc_mapping = {'V': 'v', 'arr': 'i', 'F': 'c', 'i': 'i','H': 'l', 'l-': 'l-', 'R': 'r', 'C': 'c', 'L': 'l'}

    for component in components:

        additional_flag = ''
        voltage_flag = ''
        comp_symbol = orc_mapping.get(component['OCR classification'])
        if comp_symbol == 'v':
            voltage_flag = '0 0 0'
            additional_flag += '0 0 0.5'
        elif comp_symbol in ['c','l']:
            additional_flag += '0 '
        else:
            additional_flag += '0'
        x1 = str(endpoint_coordinate_mapping[component['connections'][0]][0])
        y1 = str(endpoint_coordinate_mapping[component['connections'][0]][1])
        x2 = str(endpoint_coordinate_mapping[component['connections'][1]][0])
        y2 = str(endpoint_coordinate_mapping[component['connections'][1]][1])
        if component['value']:
            falstad_input += comp_symbol + ' ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + ' ' + str(0) + ' ' + voltage_flag + ' ' + str(component['value']) + ' ' + additional_flag + '\n'
        else:
            falstad_input += comp_symbol + ' ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + ' ' + str(0) + ' ' + voltage_flag + ' 1 ' + additional_flag + '\n'

    print(falstad_input)

    encoded_data = urllib.parse.quote(falstad_input)
    # Construct the Falstad URL
    falstad_url = f"https://www.falstad.com/circuit/circuitjs.html?cct={
        encoded_data}"
    webbrowser.open(falstad_url)


def find_connected_endpoints(skeleton, endpoints, start_id):
    visited = np.zeros_like(skeleton, dtype=bool)
    connections = [start_id]
    start_pos = endpoints[start_id]['pos']

    queue = deque([start_pos])
    # Mark start position as visited
    visited[start_pos[1], start_pos[0]] = True
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


def formatting_circuit(components, endpoints, all_connections, all_coordinates, endpoint_coordinate_mapping):
    max_end_y = 1000  # highest on page (actually lowest value)
    min_end_y = 0  # lowest on page (actually highest value)

    for group in all_connections:
        for endpoint in group:
            for point in endpoints:
                if point.get('pos')[1] > min_end_y:
                    min_end_y = point.get('pos')[1]
                if point.get('pos')[1] < max_end_y:
                    max_end_y = point.get('pos')[1]

    # determine component orientation and required height
    xaxis_component_counts = {}
    for component in components:
        if abs(x1 - x2) < 10:  # X endpoints equal
            if x1 in xaxis_component_counts:
                # count num components in each column
                xaxis_component_counts.update(
                    {x1: xaxis_component_counts.get(x1)+1})
            else:
                xaxis_component_counts.update({x1: 1})

    xaxis_high_low = {}
    for component in components:
        x1 = endpoint_coordinate_mapping[component["connections"][0]][0]
        x2 = endpoint_coordinate_mapping[component["connections"][1]][0]
        y1 = endpoint_coordinate_mapping[component["connections"][0]][1]
        y2 = endpoint_coordinate_mapping[component["connections"][1]][1]
        if y1 > y2:
            top = component["connections"][1]
            bottom = component["connections"][0]
        else:
            top = component["connections"][0]
            bottom = component["connections"][1]

        if xaxis_component_counts.get(x1) == 1:  # only for vertical components
            endpoint_coordinate_mapping.update(
                {top: (endpoint_coordinate_mapping.get(top)[0], max_end_y)})
            endpoint_coordinate_mapping.update(
                {bottom: (endpoint_coordinate_mapping.get(bottom)[0], min_end_y)})
        elif xaxis_component_counts.get(x1) and xaxis_component_counts.get(x1) > 1:
            if xaxis_high_low.get(x1):
                if endpoint_coordinate_mapping.get(top)[1] > xaxis_high_low.get(x1)[0]:
                    xaxis_high_low.update(
                        {x1: (top, xaxis_high_low.get(x1)[1])})
                elif endpoint_coordinate_mapping.get(bottom)[1] < xaxis_high_low.get(x1)[1]:
                    xaxis_high_low.update(
                        {x1: (xaxis_high_low.get(x1)[0], bottom)})
            else:
                xaxis_high_low.update({x1: (top, bottom)})
    for xcoord in xaxis_high_low.keys():
        print(xcoord, xaxis_high_low.get(xcoord))
        lowest = xaxis_high_low.get(xcoord)[0]
        highest = xaxis_high_low.get(xcoord)[1]
        endpoint_coordinate_mapping.update(
            {lowest: (endpoint_coordinate_mapping.get(lowest)[0], max_end_y)})
        endpoint_coordinate_mapping.update(
            {highest: (endpoint_coordinate_mapping.get(highest)[0], min_end_y)})
    return all_coordinates, endpoint_coordinate_mapping
