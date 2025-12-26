from setuptools import setup,find_packages

def get_requirements(file_path):
    Hypen_E_Dot = "-e ."
    requirements =[]
    
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.replace("\n","")

            if line != Hypen_E_Dot:
                requirements.append(line)
    
    return requirements
        


setup(
    name = "ML_PROJECT",
    version = '0.0.1',
    packages = find_packages(),
    author = 'Kashish',
    install_requires = get_requirements('requirements.txt')
)