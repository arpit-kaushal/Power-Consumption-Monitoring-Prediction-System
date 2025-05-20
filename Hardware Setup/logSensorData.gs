
function doGet(e) {
  Logger.log(JSON.stringify(e)); // Log the incoming request for debugging

  var result = 'Ok'; // Default response

  if (e.parameter == 'undefined') {
    result = 'No Parameters'; // Handle case where no parameters are passed
  } else {
    var sheet_id = '1lP0Zml-N00BaXpHimzrqbZ91syN4kfxaphVVYixaKQo'; // Spreadsheet ID
    var sheet = SpreadsheetApp.openById(sheet_id).getActiveSheet();
    var newRow = sheet.getLastRow() + 1; // Get the next empty row
    var rowData = []; // Array to store row data

    var Curr_Date = new Date(); // Current date
    rowData[0] = Curr_Date; // Date in column A

    var Curr_Time = Utilities.formatDate(Curr_Date, "Asia/Karachi", 'HH:mm:ss'); // Current time
    rowData[1] = Curr_Time; // Time in column B

    // Loop through all parameters in the request
    for (var param in e.parameter) {
      Logger.log('In for loop, param=' + param); // Log the parameter name
      var value = stripQuotes(e.parameter[param]); // Remove quotes from the parameter value
      Logger.log(param + ':' + value); // Log the parameter name and value

      // Assign values to the appropriate column based on the parameter name
      switch (param) {
        case 'voltage':
          rowData[2] = value; // Voltage in column C
          result = 'Voltage Written on column C';
          break;
        case 'current':
          rowData[3] = value; // Current in column D
          result += ', Current Written on column D';
          break;
        case 'power':
          rowData[4] = value; // Power in column E
          result += ', Power Written on column E';
          break;
        case 'energy':
          rowData[5] = value; // Energy (units) in column F
          result += ', Energy Written on column F';
          break;
        default:
          result = "Unsupported parameter"; // Handle unsupported parameters
      }
    }

    Logger.log(JSON.stringify(rowData)); // Log the row data for debugging

    // Write the row data to the spreadsheet
    var newRange = sheet.getRange(newRow, 1, 1, rowData.length);
    newRange.setValues([rowData]);
  }

  // Return the result as a text output
  return ContentService.createTextOutput(result);
}

// Function to remove quotes from a string
function stripQuotes(value) {
  return value.replace(/^["']|['"]$/g, "");
}
