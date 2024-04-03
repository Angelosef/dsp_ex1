# Create a virtual environment
hello1:
	python -m venv my_env
	my_env\\Scripts\\activate
	pip install -r requirements.txt

# remove virtual environment
clean:
	rmdir /s /q my_env
