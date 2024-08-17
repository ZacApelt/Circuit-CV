import easyocr
import re

# Create an OCR reader object
reader = easyocr.Reader(['en'])

def classify(result):
    components = []

    for text1 in result:
        text = text1[1]
        # Split the text where numbers stop and letters start
        split_text = re.split(r'(\d+)(\D+)', text)

        # Filter out empty strings from the result
        split_text = [part for part in split_text if part]

        # if there is an "o" or "O" in the text, replace it with a "0" and append to the end of split_text[0] to make it a number
        if "o" in split_text[1] or "O" in split_text[1]:
            split_text[0] = split_text[0] + "0"
            split_text[1] = split_text[1].replace("o", "").replace("O", "")
        
        split_text[0] = float(split_text[0])

        # fix up the units
        if "K" in split_text[1] or "k" in split_text[1]:
            split_text[0] *= 1000
        if "M" in split_text[1] or "m" in split_text[1]:
            split_text[0] *= 1000000
        if "u" in split_text[1]:
            split_text[0] *= 0.000001
        if "n" in split_text[1]:
            split_text[0] *= 0.000000001
        if "p" in split_text[1]:
            # this is for mistaking mu for p
            split_text[0] *= 0.000001
        
        # fix multiplication rounding errors
        split_text[0] = round(split_text[0], 9)

        value = split_text[0]
        text = split_text[1]

        print(text, value)

        if "V" in text or "v" in text:
            components.append({"component": "V", "value": value, "corners": text1[0]})
        elif "A" in text:
            components.append({"component": "A", "value": value, "corners": text1[0]})
        elif "H" in text:
            components.append({"component": "L", "value": value, "corners": text1[0]})
        elif "F" in text:
            components.append({"component": "C", "value": value, "corners": text1[0]})
        else:
            components.append({"component": "R", "value": value, "corners": text1[0]})
    
    return components


# Read text from an image
result = reader.readtext('./circuits/tut1.jpg')
#print(result)

classified_results = classify(result)
for component in classified_results:
    print(component)