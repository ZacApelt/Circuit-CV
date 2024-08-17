import easyocr
import re

# Create an OCR reader object
reader = easyocr.Reader(['en'])

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

def classify(result):
    components = []

    for text1 in result:
        text = text1[1]
        #print('text', text)
        if " " in text:
            value, unit = text.split(' ')
        elif not text[0].isnumeric():
            value = int(1)
            unit = text
        elif text[-1].isnumeric():
            value = int(text)
            unit = 'R'
        else:
            value, unit = split_string(text)

        value = convert_to_float_or_int(value)
        print(value, unit)

        if "K" in unit or "k" in unit:
            value *= 1000
        elif "MF" in unit or "mF" in unit or "pF" in unit: #often mistakes mu for p
            value *= 0.000001
        elif "MH" in unit or "mH" in unit or "pH" in unit: #often mistakes mu for p
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
            components.append({"component": "V", "value": value, "corners": text1[0]})
        elif "A" in unit:
            components.append({"component": "A", "value": value, "corners": text1[0]})
        elif "H" in unit:
            components.append({"component": "L", "value": value, "corners": text1[0]})
        elif "F" in unit:
            components.append({"component": "C", "value": value, "corners": text1[0]})
        else:
            components.append({"component": "R", "value": value, "corners": text1[0]})

    return components


# Read text from an image
result = reader.readtext('./circuits/cir9.png')
classified_results = classify(result)
for component in classified_results:
    print(component)
