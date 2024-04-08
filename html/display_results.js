function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }
  
  function GetURLParameter(sParam)
  {
      var sPageURL = window.location.search.substring(1);
      var sURLVariables = sPageURL.split('&');
      for (var i = 0; i < sURLVariables.length; i++) 
      {
          var sParameterName = sURLVariables[i].split('=');
          if (sParameterName[0] == sParam) 
          {
              return sParameterName[1];
          }
      }
  }
  
function load_results(url, text) {
    return function ()
    {
        $("#home").hide();
        $("#about").hide();
        $("#results").hide();

        $("#results-text").html(text);

        $.ajax({
            type: 'GET',
            url: url,
            dataType: 'json',
            success: function (data) {
                $("#main-table > tbody").empty();
                treebank_counter = 0;
                for (var treebank_id in data)
                {
                    if (data.hasOwnProperty(treebank_id))
                    {
                        treebank_counter += 1;
                        $("#main-table")
                        .find('> tbody')
                        .append(
                            $('<tr>')
                            .append($('<th>').attr('scope', 'row').text(treebank_counter))
                            .append($('<td>').text(treebank_id))
                            .append($('<td>').text(data[treebank_id]['filtered_deps_len']))
                            .append(
                                $('<td>').text(
                                    data[treebank_id]['n_yes'] + 
                                    ' (' + 
                                    (100 * data[treebank_id]['n_yes'] / data[treebank_id]['filtered_deps_len']).toFixed(2)
                                    + '%)'
                                )
                            )
                            .append(
                                $('<td>')
                                .text('show/hide')
                            )
                            .css('cursor', 'pointer')
                            .click(
                                function () {
                                    $(this).next().toggle();
                                }
                            )
                        )
                        .append(
                            $('<tr>')
                            .hide()
                            .append(
                                $('<td>')
                                .attr('colspan', '5')
                                .append(
                                    $('<table>')
                                    .addClass("table mb-0 table-hover")
                                    .bootstrapTable({
                                        columns: [
                                            {
                                                title: '#',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return index + 1;
                                                },
                                            },
                                            {
                                                title: 'Pattern',
                                                field: 'pattern',
                                            },
                                            {
                                                title: 'Occ.',
                                                field: 'n_pattern_occurence',
                                            },
                                            {
                                                title: 'Pos.',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return row["n_pattern_positive_occurence"]
                                                    + " (" + (100 * row["n_pattern_positive_occurence"] / row["n_pattern_occurence"]).toFixed(2) + "%)"
                                                    ;
                                                },
                                            },
                                            {
                                                title: 'Neg.',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return row["n_pattern_occurence"] - row["n_pattern_positive_occurence"]
                                                    + " (" + (100 * (row["n_pattern_occurence"] - row["n_pattern_positive_occurence"]) / row["n_pattern_occurence"]).toFixed(2) + "%)"
                                                    ;
                                                },
                                            },
                                            {
                                                title: 'Decision',
                                                field: 'decision',
                                            },
                                            {
                                                title: 'alpha',
                                                field: 'alpha',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(5);
                                                },
                                            },
                                            {
                                                title: 'weight',
                                                field: 'value',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(5);
                                                },
                                            },
                                            {
                                                title: 'coverage',
                                                field: 'coverage',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(2);
                                                },
                                            },
                                            {
                                                title: 'prevision',
                                                field: 'precision',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(2);
                                                },
                                            },
                                            {
                                                title: 'delta',
                                                field: 'delta',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(2);
                                                },
                                            },
                                            {
                                                title: 'g-statistics',
                                                field: 'g-statistic',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(2);
                                                },
                                            },
                                            {
                                                title: 'p-value',
                                                field: 'p-value',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(2);
                                                },
                                            },
                                            {
                                                title: 'cramers_phi',
                                                field: 'cramers_phi',
                                                formatter: function(value, row, index, field)
                                                {
                                                    return value.toFixed(2);
                                                },
                                            },
                                        ],
                                        data: data[treebank_id]['rules']
                                    })
                                )
                            )
                        );
                    }
                }
            },
            error: function (e) {
                console.log("There was an error with your request...");
                console.log("error: " + JSON.stringify(e));
                alert("error: " + JSON.stringify(e));
            }
        });

        $("#results").show();
    };
}
  
$(function() {
    $("#home-link").click(function () {
      $("#results").hide();
      $("#about").hide();
      $("#home").show();
    });
    $("#about-link").click(function () {
      $("#results").hide();
      $("#home").hide();
      $("#about").show();
    });

  $.ajax({
      type: 'GET',
      url: "./phenomena.json",
      dataType: 'json',
      contentType: "application/json",
      success: function (data)
      {
        $("#phenomena-menu").empty();
        for (key in data)
        {
            if (key.toLowerCase().includes("order"))
            {
                icon = "bi-arrow-left-right";
            }
            else if (key.toLowerCase().includes("agreement"))
            {
                icon = "icon-rotate bi-pause";
            }
            else
            {
                icon = "bi-file-earmark-spreadsheet";
            }

            $("#phenomena-menu")
			    .append(
                    $('<li>').attr("class", "nav-item")
                    .append(
                        $('<a>')
                        .click(load_results(data[key]["file"], data[key]["text"]))
                        .attr("class", "nav-link")
                        .attr("href", "#")
                        .text(" " + key)
                        .prepend($('<i>').attr("class", icon))
                    )
                )
            ;
        }
      },
      error: function (e) {
          console.log("Could not load phenomena.json");
          alert("Could not load phenomena.json");
      }
  })
  
})
  
