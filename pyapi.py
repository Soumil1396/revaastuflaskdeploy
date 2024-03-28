from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:8081"}})

# Define the folder to save response images
RESPONSE_IMAGES_FOLDER = 'response_images'
app.config['RESPONSE_IMAGES_FOLDER'] = RESPONSE_IMAGES_FOLDER

# Function to ensure the existence of the response_images folder
def create_response_images_folder():
    folder_path = os.path.join(os.getcwd(), RESPONSE_IMAGES_FOLDER)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Create the response_images folder
create_response_images_folder()

# Function to get the next available filename
def get_next_filename():
    folder_path = os.path.join(os.getcwd(), RESPONSE_IMAGES_FOLDER)
    filenames = [f for f in os.listdir(folder_path) if f.startswith('result-')]
    if not filenames:
        return 'result-01.png'
    else:
        latest_number = max(int(f.split('-')[1].split('.')[0]) for f in filenames)
        next_number = latest_number + 1
        return f'result-{next_number:02d}.png'

##def decode_base64_to_image(base64_string):
##    print("decoding base64")
##    decoded_bytes = base64.b64decode(base64_string)
##    decoded_image = np.array(Image.open(BytesIO(decoded_bytes)))
##    print("damn")
##    return decoded_image

def decode_base64_to_image4(base64_string):
    print("doing image wih 4")
    try:
        # Remove the prefix 'data:image/png;base64,'
        #print('removing the prefix')
        #print(base64_string)
        base64_string = base64_string.split(",")[0]

        print('decoding base64 to bytes')
        # Decode base64 to bytes
        decoded_data = base64.b64decode(base64_string)
        print('bytes to file')
        # Convert bytes to a file-like object
        image = BytesIO(decoded_data)

        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def decode_base64_to_image(base64_string):
    print("Decoding base64...")
    # Add padding to make the length a multiple of 4
##    base64_string += '=' * ((4 - len(base64_string) % 4) % 4)
    try:
        decoded_bytes = base64.b64decode(base64_string)
        decoded_image = np.array(Image.open(BytesIO(decoded_bytes)))
        print("Decoding successful.")
        return decoded_image
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

def fix_image_orientation(image):
    # Fix image orientation based on Exif data
    print("image", image)
    try:
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(0x0112, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Error fixing image orientation: {e}")

    return image

def find_center_and_locate(original_image_path, cropped_image_path):
##    print("got images",original_image_path, cropped_image_path, "<<<<<<PATHS")

    original_image = decode_base64_to_image(original_image_path)
    cropped_image = decode_base64_to_image(cropped_image_path)

    # Fix image orientation
##    original_image = fix_image_orientation(original_imagefin)
##    cropped_image = fix_image_orientation(cropped_imagefin)

##    print("DECODED>>>>>>>",original_image, cropped_image, "<<<<<IMAGES")

##    print("putting them both up on display")
    
    # Read the images
##    original_image = cv2.imread(original_image_path)
##    cropped_image = cv2.imread(cropped_image_path)

    if original_image is None or cropped_image is None:
        raise Exception("Error loading images.")

    print("calculating centers")

    # Find the center of the original image
    original_height, original_width, _ = original_image.shape
    center_original = (original_width // 2, original_height // 2)

    # Find the center of the cropped image
    cropped_height, cropped_width, _ = cropped_image.shape
    center_cropped = (cropped_width // 2, cropped_height // 2)

    # Find the location of the cropped image in the original image
    result = cv2.matchTemplate(original_image, cropped_image, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the center of the located region
    located_center = (max_loc[0] + center_cropped[0], max_loc[1] + center_cropped[1])
    # Draw a red circle at the located point on the original image
##    radius = 20
##    color = (0, 0, 255)  # Red color in BGR
##    thickness = 20  # Filled circle
##    newog = cv2.circle(original_image, located_center, radius, color, thickness)
##    

    # Display the images using Matplotlib
##    plt.subplot(121), plt.imshow(newog)
##    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##    # Display the images using Matplotlib
##    plt.imshow(original_image)
##    plt.title('Original Image')
####    plt.gca().set_aspect('auto', djustable='box')
##    plt.xticks([]), plt.yticks([])
##    plt.show()
##
##    plt.subplot(122), plt.imshow(cropped_image)
##    plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])
##
##    plt.show()
    
    return center_original, center_cropped, located_center, original_image, cropped_image


@app.route("/")
def home():
    return "Revaastu Flask Server Running."

@app.route('/process_images', methods=['POST'])
def process_imageses():
    # Get file paths from the request
    original_path = request.form['original_image']
    cropped_path = request.form['cropped_image']
    
    # Decode base64 strings into image data
    original_image_data = base64.b64decode(original_path)
    cropped_image_data = base64.b64decode(cropped_path)

    # Open images using OpenCV
    original_image = cv2.imdecode(np.frombuffer(original_image_data, np.uint8), cv2.IMREAD_COLOR)
    cropped_image = cv2.imdecode(np.frombuffer(cropped_image_data, np.uint8), cv2.IMREAD_COLOR)

    # Find the center of the cropped image
    cropped_center = (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)

    # Find the location of the cropped image in the original image
    result = cv2.matchTemplate(original_image, cropped_image, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the center of the located region
    located_center = (max_loc[0] + cropped_center[0], max_loc[1] + cropped_center[1])

    # Mark the center of the cropped image on the original image with a red circle
    marked_image = original_image.copy()
    radius = 20
    color = (0, 0, 128)  # Maroon color in BGR
    thickness = -1  # Filled circle
    marked_image = cv2.circle(marked_image, located_center, radius, color, thickness)
    # Save the marked image temporarily
    cv2.imwrite('marked_image.png', marked_image)
    # Convert the marked image to base64
    _, encoded_image = cv2.imencode('.png', marked_image)
    marked_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
    print("encoded")
    return jsonify({ "marked_image_base64": marked_image_base64})

def overlay_images_on_cropped_region(main_image, top_image, crop_position):
    print("cropped positions", crop_position)
    # Convert the main and top images to RGBA mode
    main_image = main_image.convert("RGBA")
    top_image = top_image.convert("RGBA")

    # Resize the top image to match the dimensions of the cropped region
    cropped_region = main_image.crop(crop_position)
    top_image = top_image.resize(cropped_region.size)

    # Calculate the position to paste the top image on the border of the cropped region
    paste_position = (crop_position[0], crop_position[1])

    # Composite the images
    result = Image.new("RGBA", main_image.size, (0, 0, 0, 0))
    result.paste(main_image, (0, 0))
    result.paste(top_image, paste_position, top_image)

    return result

@app.route('/tropinmain', methods=['POST'])
def tropinmain():
    try:
        print('Overlaying')

        # Retrieve images from request.form
        top_base64 = request.form['top']
        main_base64 = request.form['main']
        crop_base64 = request.form['crop']

        print("Decoding images")

        # Decode base64 strings to images
        top_image = Image.fromarray(decode_base64_to_image(top_base64))
        main_image = Image.fromarray(decode_base64_to_image(main_base64))
        cropped_image = Image.fromarray(decode_base64_to_image(crop_base64))

        # top_image.show()
        # main_image.show() #orientation fix
        print("fixingOrientation")
        # otc = fix_image_orientation(main_image) 
        # otm = main_image.rotate(270, expand=True)
        if main_image.size[0] > main_image.size[1]:
            print("This is expected to fail. But program to handle it is working, so hold tight!!")
            otm = main_image.rotate(270, expand=True)
        else:
            print("This is successful. No worries!!")
            otm = fix_image_orientation(main_image)
        print("opening otm")
        # otm.show()
        # cropped_image.show()
        print("cropped dimensions", cropped_image, cropped_image.width, cropped_image.height, cropped_image.size )

        # otc.show()

        # Find the center of the cropped image
        cropped_width, cropped_height = cropped_image.size
        center_cropped = (cropped_width // 2, cropped_height // 2)



        # Perform template matching to find the location of the cropped image in the original image
        result = cv2.matchTemplate(np.array(otm), np.array(cropped_image), cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # # Create a mask to color the region black
        # mask = Image.new('L', main_image.size, 0)
        # mask.paste(255, (max_loc[0], max_loc[1], max_loc[0] + cropped_image.width, max_loc[1] + cropped_image.height))

        # # Apply the mask to the main image
        # main_image.paste((0, 0, 0, 255), mask=mask)

        # Overlay the top image on the borders of the cropped image in the main image
        result_image = overlay_images_on_cropped_region(otm, top_image, (max_loc[0], max_loc[1], max_loc[0] + cropped_image.width, max_loc[1] + cropped_image.height))
        # result_image.show()

         # Save the result image with a unique filename
        result_image_filename = get_next_filename()
        result_image_path = os.path.join(app.config['RESPONSE_IMAGES_FOLDER'], result_image_filename)
        result_image.save(result_image_path)

        # Encode the result image to base64
        # result_base64 = encode_image_to_base64(result_image)

        return jsonify({ "fileName": result_image_filename})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
# Serve the saved images using send_file
@app.route('/get_result_image/<filename>')
def get_result_image(filename):
    try:
        # Construct the full path to the result image
        result_image_path = os.path.join(app.config['RESPONSE_IMAGES_FOLDER'], filename)

        # Return the result image using send_file
        return send_file(result_image_path, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/apply-over', methods=['POST'])
def overlay_images():
##    try:
        print('Overlaying')
        top = request.form['top']
        main = request.form['main']
##        print('base64files', top, main, "<<<<<<base")

        print("images to decode")

        decoded_top = decode_base64_to_image(top)
        decoded_main = decode_base64_to_image(main)

        print("decoded images")

        if decoded_top is None or decoded_main is None:
            raise Exception("Error loading images.")
        
##        print("Applying overlay", decoded_top, "||||||||||||||||", decoded_main)

        # Open the decoded images
##        tp = Image.open(decoded_top)
##        mn = Image.open(decoded_main)
##
##        print("tp&mn", tp, mn)
##        # Convert the NumPy array to a PIL Image
        top_image = Image.fromarray(decoded_top)
        main_image = Image.fromarray(decoded_main)

        otc = fix_image_orientation(main_image)     

        # Overlay the images
        print("OVERlaYING NOW!!")
##        result = overlay_images_function(mn, tp)
        result = overlay_images_function(otc, top_image)
##        print("done", result)
        image = result.rotate(270, expand=True)
        image.show()

        # Encode the result to base64
        result_base64 = encode_image_to_base64(image)
##        print(result_base64, "this here overlay image")

        return jsonify({'result': result_base64})
##    except Exception as e:
##        print(f"Error overlaying images: {e}")
##        return jsonify({'error': 'Error overlaying images'}), 500

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        # Get text values
        text1 = request.form['text1']
        text2 = request.form['text2']
##        cropped_image = request.files['cropped_image']

        # Process text values
        result = {
            'message': 'Text processed successfully',
            'text1': text1,
            'text2': text2,
##            'image': cropped_image,
        }

        # Display the processed information
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



@app.route('/process_imagesmain', methods=['POST'])
def process_images():
    print("hit")
    # Get file paths from the request
    original_path = request.form['original_image']
    cropped_path = request.form['cropped_image']
    compass_path = request.form['compass_image']
    angle = request.form['angle']
    
    # Decode base64 strings into image data
    original_image_data = base64.b64decode(original_path)
    cropped_image_data = base64.b64decode(cropped_path)
    compass_image_data = base64.b64decode(compass_path)
    print("decoded")

    # Open images using Pillow
    original_image = Image.open(BytesIO(original_image_data))
    cropped_image = Image.open(BytesIO(cropped_image_data))
    compass_image = Image.open(BytesIO(compass_image_data))

    print("fixing image orientation")

     # Fix image orientation
    original_imagefin = fix_image_orientation(original_image)
    cropped_imagefin = fix_image_orientation(cropped_image)
    compass_imagefin = fix_image_orientation(compass_image)

    print("open pillow")

    # Display the images (for demonstration purposes)
##    original_imagefin.show()
##    cropped_imagefin.show()
##    compass_imagefin.show()

    print("goin for the centers")

    # Find the center of the original image
    original_width, original_height = original_imagefin.size
    center_original = (original_width // 2, original_height // 2)

    # Find the center of the cropped image
    cropped_width, cropped_height = cropped_imagefin.size
    center_cropped = (cropped_width // 2, cropped_height // 2)

    # find the location of the cropped image in the original image
    result = cv2.matchTemplate(np.array(original_imagefin), np.array(cropped_imagefin), cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
##    print('res>>', result)

    # Calculate the center of the located region
    located_center = (max_loc[0] + center_cropped[0], max_loc[1] + center_cropped[1])

    # Mark the located center with a circular dot on the original image
##    draw = ImageDraw.Draw(original_imagefin)
##    dot_radius = 16
##    draw.ellipse((located_center[0] - dot_radius, located_center[1] - dot_radius, located_center[0] + dot_radius, located_center[1] + dot_radius), fill="red")

    # Ensure the compass image's border does not go out of the original image size
    compass_width, compass_height = compass_imagefin.size
    center_compass = (compass_width // 2, compass_height // 2)
    print('centerCOmpass', center_compass)
    
    # Mark the located center with a circular dot on the compass image
##    draw = ImageDraw.Draw(compass_imagefin)
##    dot_radius = 16
##    draw.ellipse((center_compass[0] - dot_radius, center_compass[1] - dot_radius, center_compass[0] + dot_radius, center_compass[1] + dot_radius), fill="red")

     # Fix the width and height of the compass image
    fixed_compass_width, fixed_compass_height = 1024, 1024

    # Ensure the compass image's border does not go out of the original image size
    compass_position = (
        max(0, min(center_compass[0] - fixed_compass_width // 2, original_width - fixed_compass_width)),
        max(0, min(center_compass[1] - fixed_compass_height // 2, original_height - fixed_compass_height))
    )

    print('compassPosition', compass_position)

    # Resize the compass image to the fixed width and height
    compass_imagefin = compass_imagefin.resize((fixed_compass_width, fixed_compass_height), Image.BICUBIC)
    # compass_imagefin = compass_imagefin.resize((cropped_width, cropped_height), Image.BICUBIC)
    print("angle>>",angle)
    # Convert the angle to a float
    angle = float(angle)
    angle = -angle
    compass_imagefin = compass_imagefin.rotate(angle, expand=True)

    # Calculate the position to place the center of the compass image on the located center of the cropped image
    compass_position = (
        located_center[0] - compass_imagefin.width // 2,
        located_center[1] - compass_imagefin.height // 2
    )

    # Paste the resized compass image onto the original image
    original_imagefin.paste(compass_imagefin, compass_position, compass_imagefin)

    print("OGPLAYA", original_imagefin)

    # Display the updated image
    # original_imagefin.show()

    # Save the result image with a unique filename
    result_image_filename = get_next_filename()
    result_image_path = os.path.join(app.config['RESPONSE_IMAGES_FOLDER'], result_image_filename)
    original_imagefin.save(result_image_path)

    # Encode the result to base64
    # result_base64 = encode_image_to_base64(original_imagefin)

##    print('sizew', compass_width, 'h', compass_height)
##    
##    compass_position = (
##        max(0, min(center_compass[0] - fixed_compass_width // 2, original_width - fixed_compass_width)),
##        max(0, min(center_compass[1] - fixed_compass_height // 2, original_height - fixed_compass_height))
##    )
##
##    # Paste the compass image onto the original image
##    original_imagefin.paste(compass_imagefin, compass_position, compass_imagefin)
##
##    # Display the updated image
##    print('opening final image found center and placed compass ', result_base64)
##    original_imagefin.show()

##    return jsonify(located_center)
    return jsonify({ "fileName": result_image_filename})



@app.route('/check_server', methods=['GET'])
def check_server():
    return jsonify({'message': 'hi'})

def encode_image_to_base64(image):
    print("encoding")
    try:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def encode_image_to_base641(image):
    print("encoding")
    # Encode the image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue().decode('utf-8')

def overlay_images_function(main_image, top_image):
    print("overlayiing finale")
    # Resize the top image to match the dimensions of the main image
    top_image = top_image.resize(main_image.size)

    # Composite the images
    result = Image.alpha_composite(main_image.convert("RGBA"), top_image.convert("RGBA"))
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run(port=33507)
