import os, cv2, pickle
from window import get_window

save_path = os.path.join('../resources/screen/hashes.pickle')

window = get_window()
image = cv2.resize(window.screenshot(), (1920, 1080), interpolation=cv2.INTER_AREA)

labels = ['main_menu', 'level_select', 'level_exit', 'level_generator', 'user_levels', 'options'] #'In Level'

if os.path.isfile(save_path):
    with open(save_path, 'rb') as file:
        existing_images, _ = pickle.load(file)
        images = existing_images + [image]
else:
    images = [image]

with open(save_path, 'wb') as file:
    pickle.dump((images, labels), file, protocol=pickle.HIGHEST_PROTOCOL)

