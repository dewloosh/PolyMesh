import os
import glob


def generate_notebook_gallery_rst(title:str = "Examples", reversed:bool=False):
    
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    gallery_rst_path = os.path.join(folder_path, 'examples_gallery.rst')
    gallery_folder_path = os.path.join(folder_path, 'examples')
    notebook_list = glob.glob(os.path.join(gallery_folder_path, '*.ipynb'))
    notebook_names = [os.path.basename(file_path) for file_path in notebook_list]
    
    with open(gallery_rst_path, 'w') as f:
        f.write(title + "\n")
        f.write(len(title) * "-" + "\n")
        f.write("\n")
        f.write(r".. nbgallery::" + "\n")
        f.write(r"    :maxdepth: 1" + "\n")
        if reversed:
            f.write(r"    :reversed:" + "\n")
        f.write("\n")
        
        for nb in notebook_names:
            f.write(r"    examples/" + nb + "\n")
        

if __name__=='__main__':
    generate_notebook_gallery_rst(reversed=True)