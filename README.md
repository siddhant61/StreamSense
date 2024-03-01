# StreamSense (WIP)

## Introduction

StreamSense is an innovative open-source platform tailored for real-time acquisition, processing, and visual representation of physiological data. It seamlessly interfaces with Muse and Empatica E4 sensors, offering a robust solution for researchers and practitioners interested in biofeedback, cognitive load assessment, and physiological monitoring.

## Key Features

- **Multi-Device Support**: Compatible with Muse and Empatica E4 sensors.
- **Versatile Data Streams**: Handles various data types including EEG, BVP, ACC, GSR, and PPG.
  ![Data Streams 1](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide22.JPG)
  ![Data Streams 2](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide23.JPG)
- **Real-Time Visualization**: Offers tools for immediate data visualization to monitor ongoing sessions.
  ![Data Visualization 1](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide26.JPG)
  ![Data Visualization 2](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide27.JPG)
- **Event Logging**: Allows users to mark events in real-time, facilitating data analysis.
- **Data Recording**: Captures data streams in formats suitable for detailed offline analysis.
  ![Signal Monitoring](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide24.JPG)
  ![Data Quality](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide25.JPG)

## Installation

Clone the repository:
```
git clone https://github.com/siddhant61/StreamSense.git
```
Navigate to the StreamSense directory and install dependencies:
```
cd StreamSense
pip install -r requirements.txt
```
## Quickstart Guide
Running StreamSense
To start the StreamSense application, run:
```
python main.py
```
Commands
- Menu Navigation: --command menu to access the main menu.
- Data Streaming: --command stream followed by --dev [device] to start streaming.
- Data Viewing: --command view followed by --data [data_stream] to view data.
- Recording: --command record to start recording the data streams.
- Event Logging: --command logger to open the event logger console.
- For more information on commands, use:
```
python main.py --help
```
![CLI](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide21.JPG)

## System Architecture
![System Architecture](https://github.com/siddhant61/StreamSense/blob/master/Logs/Slide20.JPG)

The architecture diagram provides a visual representation of the StreamSense workflow, illustrating the integration between the Command Line Interface, sensor devices, and various system components.

## Research Methodology
StreamSense is utilized in a structured research methodology aimed at correlating physiological signals with mental workload and stress levels. The methodology.docx document in the repository outlines the experimental design, data privacy considerations, and ethical compliance.

## Documentation
The docs folder contains detailed documentation for each component and usage guidelines.

## Contributing
Contributions to StreamSense are highly appreciated. Whether it be feature requests, bug reports, or code contributions, please refer to CONTRIBUTING.md for guidance on how to contribute effectively.

## License
StreamSense is distributed under the MIT License. See LICENSE for more information.

## Acknowledgments
Our gratitude goes to all the researchers and users who have made StreamSense a valuable tool in the pursuit of advancing physiological research.

