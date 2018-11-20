<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta name="description" content="Bottle web project template">
	<meta name="author" content="datamate">
	<link rel="icon" href="/static/favicon.ico">		
	<title>Project</title>
	<!--<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">-->
    <link rel="stylesheet" href="timeline.css" type="text/css" >
    <script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/knockout/3.4.2/knockout-min.js"></script>
    
    <script>
    $(document).ready(function() {
        function Row(term, body, subject) {
            this.body = ko.observable(body.replace(term, "<mark>$&</mark>"));
            this.subject = ko.observable(subject);
        }
        // This is a simple *viewmodel* - JavaScript that defines the data and behavior of your UI
        function AppViewModel() {
            this.searchTerm = ko.observable("");
            this.searchResults = ko.observableArray([]);
            var that = this;
            
            this.searchTerm.subscribe(function (newSearch) {
                $.get('get_posts', {searchTerm: newSearch})
                .done(function(data) {
                    data = JSON.parse(data);
                    var newResults = ko.utils.arrayMap(data, function(item) {
                        return new Row(newSearch, item.body, item.subject);
                    });
                    that.searchResults.removeAll()
                    ko.utils.arrayPushAll(that.searchResults, newResults);
                })
                .fail(function() {
                    console.error("Failure: "+newSearch);
                })
            });
        }

        // Activates knockout.js
        ko.applyBindings(new AppViewModel());
    });
    </script>
</head>
<body>
	<!-- Static navbar -->
    <div class="timeline">
    % for row in rows:
        <div class="container right">
            <div class="content">
                <h2>{{ row['subject'].title() }}</h2>
                <sup>{{ row['sent'][:10] }}</sup>
                <pre>{{ row['body'] }}</pre>
            </div>
        </div>
    % end
    </div>
	<!-- /container -->
</body>
</html>