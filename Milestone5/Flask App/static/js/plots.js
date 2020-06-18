$('#country').on('change',function(){
	
    $.ajax({
        url: "/country",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'selected': document.getElementById('country').value		
        },
        dataType:"json",
        success: function (x) {
            Plotly.newPlot('overallCase', x.graphs[0] );			
			Plotly.newPlot('activeCase', x.graphs[1] );
			Plotly.newPlot('svmPred', x.figs[0] );			
			Plotly.newPlot('polyPred', x.figs[1] );
			Plotly.newPlot('mlpPred', x.figs[2] );
        }
    });
})
