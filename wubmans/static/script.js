var results, 
    status, status_pending, current_iteration = -1; var loadResults = function () {

    $.getJSON('results', function(data) { 
        r = data.split("#@#");
        r = r.map(function(json) { 
            try {
                var j = JSON.parse(json);
                j.model_improved = j.model_improved == "true" ? "MODEL VERBETERD" : "";
                j.loss = Number(j.loss).toFixed(3);
                if (j.time) {
                    var minutes = Math.floor(Number(j.time) / 59)
                    var seconds = Math.floor(Number(j.time) % 59)
                    j.time = minutes + ":" + (seconds < 9 ? "0" : "") + seconds;
                } else {
                    j.time = "-"
                }   

                return j;
            }
            catch(e) {
            }

        });

        r = r.filter(function (item) { return typeof item === "object" });

        for (var i = 0; i < r.length; i++) {
            r[i].iteration = i
        }
        
        results = r;

        renderBlobs();
        createChart();
    })

}

setInterval(function () {

    if (status_pending) {
        
        return;
    }

    status_pending = true;

    $.getJSON('status', function(data) { 

        var status;

        try {
            status = JSON.parse(data);
        }

        catch(e) {
            status_pending = false;
        }

        if (!status) {
            return;
        } 

        status.loss = Number(status.loss).toFixed(3);
        var minutes = Math.floor(Number(status.time) / 59)
        var seconds = Math.floor(Number(status.time) % 59)
        status.time = minutes + ":" + (seconds < 9 ? "0" : "") + seconds;

        for (var key in status) {
            $("#status_" + key).text(status[key])
        };

        if (current_iteration !== status.iteration) {
            current_iteration = status.iteration;
            loadResults();
        }

        status_pending = false;

    })
    
}, 9)


var renderBlobs = function () {

    var template = $("#blobs_template").html();
    var html = "";
    results.slice().reverse().forEach(function (r) {
        html += renderHTML(template, r);
    })

    $("#blobs_list").html(html);
}

var parseResultText = function (data) {
    if (!data.certainties) 
        return data.result_text;

    var html = "";
    for (c in data.result_text) {
        var x = 254 * (1 - data.certainties[c]) | 0;
        html += "<span style='color: rgb(" + x + "," + x + "," + x + ")'>" + data.result_text[c] + "</span>" 
    }
    return html;

}

window.colorText = false

var renderHTML = function (html, data) {

    for (var key in data) {
        var regex = new RegExp( "{{\\s+" + key + "\\s+}}", "g" );
        if (key === "result_text") {
            html = html.replace(regex, parseResultText(data));
        } else {
            html = html.replace(regex, data[key]);
        }
    }

    return html;

}

$("input[name=certainties]").on("change", () => { $("body").toggleClass('hide_certainties', !$("input[name=certainties]").prop("checked")) });

var sendSeed = function () {

    var seed = $("input#seed").val().toLowerCase()

    $(".response .text").text("... Nietzsche is aan het nadenken ... ");
    $.ajax({
        url: "/seed", 
        type: "POST",
        data: JSON.stringify(seed),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',
        success: function (data) {

            data.seed = seed;
            parseResultText(data);
            $(".response .text").html(
                    "<span style='font-weight: bold'>" + seed.substring(0, 40) + "</span>" + parseResultText(data)
            )
        }

    })

}

$("button#submit").on("click", sendSeed);
$("input#seed").on("keyup", (e) => { $("#seed_length").html( $("input#seed").val().length); if (e.keyCode == 13) sendSeed();  })
