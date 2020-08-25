html_file = open('results.html', 'wb')

message = """<html>
<header>
    <p style="text-align: center;"><img src="logo.png"></p>
    <h2 style="text-align: center; font-size: 45px;">Results</h2>
</header>
<body>
    <p style="text-align: center;"><img src="BiasDetectionHistogram.png"></p>
</body>
</html>"""

html_file.write(message)
html_file.close()