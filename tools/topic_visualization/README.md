# Topics tree visualization

Tree visualization done with [d3.js](https://d3js.org).

Steps to visualize topics tree:
* Verify path to json file with topics tree, variable `json_path` in [html](topic_visualization_2022.html) file. Defaults is set to automatic evaluation topics tree.
* Disable Cross-Origin Restrictions in your browser.
  * Safari: check `Disable Cross-Origin Restrictions` in Develop menu.
  * Chrome: start Chrome with the follwing command: `open -n -a /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --args --user-data-dir="/tmp/chrome_dev_test" --disable-web-security`
  * Firefox: go to `about:config`, set `security.fileuri.strict_origin_policy` to false.
* Open [html file](topic_visualization_2022.html) in your browser.
* Select the topic number you want to display with the input on the top left corner.

