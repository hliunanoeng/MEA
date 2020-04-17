The MEA consists of multiple scripts for rudimentary MEA recording analysis and visualization; the user may use it to gauge and compare the performance of different Igloo designs.

General work flow:
1. Export the recording files using MultiChannelDataManager to .h5 format
2. Select the recording interval one wishes to analyze using selectTimeSequence from Util.py
3. Generate a summary review of the MEA performance using AnalyzeRecordings from SpikeAnalysis.py
4. Visualize the analysis result using functions from SpikeVisualization.py
5. Test the statistic significance of the performance difference between electrode using Ttable from SpikeTest.py

If you have further questions, please refer to the code samples or contact me at hliunanoeng@gmail.com.

Han, 03 03 2019.
