 $(function () {
$('#id_label_name').autocomplete({
        delay: 500,
				minLength: 2,
				source: function(request, response) {
                    var domainId = $("#id_domain").val();

					$.getJSON("/get_autocomplete_labels/"+domainId, {
						s: request.term
					}, function(data) {
					    console.log(data)
						// data is an array of objects and must be transformed for autocomplete to use
						var array = data.Response === "False" ? [] : $.map(data, function(m) {
							return {
								label: m.name + " ",
								id:  m.id
							};
						});
					    if ( array.length == 0) {
					        $('#id_label').val('');
                            array.push({
								label: 'Nu exista intentul',
								id: ''
							})
                        }
						response(array);
					});
				},
				focus: function(event, ui) {
					// prevent autocomplete from updating the textbox
					event.preventDefault();
					$('#id_label').val(ui.item.id);
				},
				select: function(event, ui) {
					// prevent autocomplete from updating the textbox
					event.preventDefault();
					// navigate to the selected item's url
                    $('#id_label_name').val(ui.item.label);
					$('#id_label').val(ui.item.id);
				}
			});
        $(".btnDel").click(function (e) {
            e.preventDefault();
            var chat_id = $(this).data("chat_id");
            var msg_id = $(this).data("message_id");
            console.log(msg_id);
            deleteMessage(msg_id);
        });
        $(".btnAdd").click(function (e) {
            e.preventDefault();
            $("#popup").show();
            $("#id_parent").val($(this).data("parent_id"));
            $("#id_chat").val($(this).data("chat_id"));
            var msg_id = $(this).data("message_id");
            if (parseInt(msg_id) > 0){
                getMessage(msg_id);
            }

        });
        $("#close").click(function(e){
            e.preventDefault();
            $("#popup").hide();
        });

        $("#id_cancel_delete").click(function(e){
            e.preventDefault();
            $("#popupConfirmation").hide();
        });

        $("#id_confirmation_delete").click(function(e){
            e.preventDefault();
            var msg_id = $("#id_confirmation_delete").data("message_id");
            console.log(msg_id);
            deleteMessage(msg_id, true);
        });


    // Submit post on submit
    $('#post-form').on('submit', function(e){
        e.preventDefault();
        console.log("form submitted!")  // sanity check
        var msg_id = $('#id_message_id').val();
        create_update_message(msg_id);
    });
        function deleteMessage(msg_id, confirm=false) {
            $.ajax({
                url : "/delete_message/"+msg_id, // the endpoint
                type : "GET", // http method
                data : { 'confirm':confirm}, // data sent with the post request
                // handle a successful response
                success : function(json) {
                    console.log(json);
                    console.log("success");
                    if (json.success == true){
                        if (json.confirmation == true){
                            if (json.deleted == true){
                                location.reload();
                            }
                            $('#txtConfirmation').html(json.message);
                            $("#popupConfirmation").show();
                            $("#id_confirmation_delete").data("message_id",msg_id); //

                        }
                        //location.reload();
                    }else{
                        alert(json.error);
                    }// another sanity check
                },
                // handle a non-successful response
                error : function(xhr,errmsg,err) {
                    $('#results').html("<div class='alert-box alert radius' data-alert>Oops! We have encountered an error: "+errmsg+
                        " <a href='#' class='close'>&times;</a></div>"); // add the error to the dom
                    console.log(xhr.status + ": " + xhr.responseText); // provide a bit more info about the error to the console
                }
            });
        }
    function getMessage(msg_id) {
        $.ajax({
            url : "/get_message/"+msg_id, // the endpoint
            type : "GET", // http method
            data : { }, // data sent with the post request
            // handle a successful response
            success : function(json) {
                console.log(json); // log the returned json to the console
                $('#id_parent').val(json.parent_id);
                $('#id_chat').val(json.chat_id);
                $('#id_label').val(json.label_id);
                $('#id_label_name').val(json.label_name);
                $('#id_message').val(json.message);
                $('#id_message_id').val(msg_id);
                if (json.human == false){
                    $('#id_label_name').prop("readonly", true);
                    $('#id_human').val('0');
                }else{

                    $('#id_human').val('1');
                }
                console.log("success"); // another sanity check
            },
            // handle a non-successful response
            error : function(xhr,errmsg,err) {
                $('#results').html("<div class='alert-box alert radius' data-alert>Oops! We have encountered an error: "+errmsg+
                    " <a href='#' class='close'>&times;</a></div>"); // add the error to the dom
                console.log(xhr.status + ": " + xhr.responseText); // provide a bit more info about the error to the console
            }
        });
    }
    // AJAX for posting
    function create_update_message(id) {
        console.log("create post is working!") // sanity check

        post_url = "/create_message/";

        if (parseInt(id) > 0){
            post_url = "/update_message/"+id;
        }

        $.ajax({
            url : post_url, // the endpoint
            type : "POST", // http method
            data : { parent : $('#id_parent').val(),
                    chat : $('#id_chat').val(),
                    label : $('#id_label').val(),
                    message : $('#id_message').val(),
                    domain : $('#id_domain').val(),
                    label_name : $('#id_label_name').val(),
                    human: $('#id_human').val()
            }, // data sent with the post request
            // handle a successful response
            success : function(json) {
                $('#id_parent').val('');
                $('#id_chat').val('');
                $('#id_label').val('');
                $('#id_message').val(''); // remove the value from the input
                $('#id_label_name').val('');
                $("#popup").hide();
                location.reload();
                console.log(json); // log the returned json to the console
                //$("#talk").prepend("<li><strong>"+json.text+"</strong> - <em> "+json.author+"</em> - <span> "+json.created+
                //    "</span> - <a id='delete-post-"+json.postpk+"'>delete me</a></li>");
                console.log("success"); // another sanity check
            },
            // handle a non-successful response
            error : function(xhr,errmsg,err) {
                $('#results').html("<div class='alert-box alert radius' data-alert>Oops! We have encountered an error: "+errmsg+
                    " <a href='#' class='close'>&times;</a></div>"); // add the error to the dom
                console.log(xhr.status + ": " + xhr.responseText); // provide a bit more info about the error to the console
            }
        });
    };


    // This function gets cookie with a given name
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie != '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) == (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');

    /*
    The functions below will create a header with csrftoken
    */

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    function sameOrigin(url) {
        // test that a given url is a same-origin URL
        // url could be relative or scheme relative or absolute
        var host = document.location.host; // host + port
        var protocol = document.location.protocol;
        var sr_origin = '//' + host;
        var origin = protocol + sr_origin;
        // Allow absolute or scheme relative URLs to same origin
        return (url == origin || url.slice(0, origin.length + 1) == origin + '/') ||
            (url == sr_origin || url.slice(0, sr_origin.length + 1) == sr_origin + '/') ||
            // or any other URL that isn't scheme relative or absolute i.e relative.
            !(/^(\/\/|http:|https:).*/.test(url));
    }

    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && sameOrigin(settings.url)) {
                // Send the token to same-origin, relative URLs only.
                // Send the token only if the method warrants CSRF protection
                // Using the CSRFToken value acquired earlier
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

    /*var oldCount = 0;
    function  addRow(parrent_id) {
        var markup = "<tr><td id='line-new'><input type='text' id='id_label' name='label'></td>" +
            "<td><input type='text' id='id_text' name='text'></td>" +
            "<td><span class=\"badge badge-primary myBtnSave\" data-parent=\"0\" data-chat=\"1\">Add</span></td></tr>";
        $("table tbody").append(markup);
    }

    function delRow(lineNo){
        $("table tbody").find('#line-'+lineNo).each(function(){

            $(this).parents("tr").remove();

        });
    }

    $("#btnAdd").click(function (e) {
        e.preventDefault();
        if(!$("#line-new").length){
            addRow(1);
        }

    });
    $('body').on('click', 'span.myBtnSave', function () {
        console.log("a");
        $('input[id^="id_label"]').each(function(input){
            var value = $('input[id^="id_label"]').val();
            var id = $('input[id^="id_label"]').attr('id');
            console.log('id: ' + id + ' value:' + value);
        });
        $('input[id^="id_text"]').each(function(input){
            var value = $('input[id^="id_text"]').val();
            var id = $('input[id^="id_text"]').attr('id');
            console.log('id: ' + id + ' value:' + value);
        });
    })*/
});