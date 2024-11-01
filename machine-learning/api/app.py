from flask import Flask, request, jsonify
import requests
import json
import threading

app = Flask(__name__)

# Function to simulate running the machine learning task in a separate thread
def run_ml_task(task_id, api_callback_url):
    try:
        # Simulate calling a machine learning model
        ml_detection_status = "Clean"  # Example detection status, e.g., from ML model output
        # TODO: Integrate actual ML model here to set ml_detection_status based on task results

        # Send status after completion of the malware detection task
        response = requests.post(api_callback_url, json={
            "task_id": task_id, 
            "detection": ml_detection_status, 
            "status": "success"
        })
        
        if response.status_code == 200:
            print(f"Callback successful for task {task_id}.")
        else:
            print(f"Callback failed for task {task_id}: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error in ML task callback for task {task_id}: {str(e)}")

@app.route('/callback', methods=['POST'])
def ml_task_status():
    # Get JSON data from the request body as a string and load it
    data = request.data.decode('utf-8')
    try:
        data_dict = json.loads(data)  # Convert JSON string to a dictionary
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    
    # Check if 'task_id' is present in the data
    if 'task_id' not in data_dict:
        return jsonify({"error": "task_id is required"}), 400
    
    task_id = data_dict.get('task_id')
    
    # Build the report URL to access task status
    report_url = f"http://localhost:8000/apiv2/tasks/get/report/{task_id}/"
    api_callback_url = "http://localhost:5000/api/callback"
    
    try:
        # Send a GET request to the report endpoint to fetch the status
        response = requests.get(report_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            filename = f"report_{task_id}.json"
            
            # Save the report to a file
            with open(filename, 'wb') as file:
                file.write(response.content)

            # Start a new thread to run the machine learning task
            ml_thread = threading.Thread(target=run_ml_task, args=(task_id, api_callback_url))
            ml_thread.start()

            return jsonify({"task_id": task_id, "status": "Processing started"}), 200
        else:
            # If report retrieval failed, send failure status
            requests.post(api_callback_url, json={
                "task_id": task_id, 
                "detection": "Unknown", 
                "status": "failed"
            })
            return jsonify({"error": f"Failed to retrieve report for task_id {task_id}"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to report endpoint failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000)