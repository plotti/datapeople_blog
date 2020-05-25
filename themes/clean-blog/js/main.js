(function ($) {
    "use strict";

    /*----------------------
    Document-Reader-Function
    -----------------------*/
    $(document).on('ready', function () {

        /*--------------------
        Show-Body
        ----------------------*/
        $('body').css({
            'opacity': '1'
        });

        /*---------------------
        Scroll-Top-Button
        ----------------------*/
        $.scrollUp({
            scrollText: '<i class="fa fa-angle-up"></i>',
            easingType: 'linear',
            scrollSpeed: 900,
            animation: 'fade'
        });

        /*---------------------
        Data-Lity-Lightbox
        ----------------------*/
        $(document).on('click', '[data-lightbox]', lity);
        /*----------------------
        Product-Slider-Carousel
        -----------------------*/
        $('.product-slide').owlCarousel({
            loop: true,
            margin: 0,
            responsiveClass: true,
            nav: false,
            autoplay: true,
            autoplayTimeout: 4000,
            smartSpeed: 1000,
            navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right" ></i>'],
            responsive: {
                0: {
                    items: 1,
                },
                480: {
                    items: 2
                },
                768: {
                    items: 3
                },
                1000: {
                    items: 4
                }
            }
        });

        /*----------------------
        Product-Slider-Carousel
        -----------------------*/
        $('.product-single-slider').owlCarousel({
            loop: true,
            margin: 0,
            responsiveClass: true,
            nav: false,
            autoplay: true,
            autoplayTimeout: 2000,
            smartSpeed: 1000,
            animateIn: 'fadeIn',
            animateOut: 'fadeOut',
            navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right" ></i>'],
            responsive: {
                0: {
                    items: 1,
                },
                480: {
                    items: 1
                },
                768: {
                    items: 1
                },
                1000: {
                    items: 1
                }
            }
        });

        /*----------------------
        Product-Slider-Carousel
        -----------------------*/
        $('.product-slider-2').owlCarousel({
            loop: true,
            margin: 0,
            responsiveClass: true,
            nav: false,
            autoplay: true,
            autoplayTimeout: 4000,
            smartSpeed: 1000,
            navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right" ></i>'],
            responsive: {
                0: {
                    items: 1,
                },
                480: {
                    items: 1
                },
                768: {
                    items: 1
                },
                1000: {
                    items: 1
                }
            }
        });

        /*----------------------
        Product-Slider-Carousel
        -----------------------*/
        $('.testimonials').owlCarousel({
            loop: true,
            margin: 0,
            responsiveClass: true,
            nav: false,
            autoplay: true,
            mouseDrag: false,
            touchDrag: false,
            autoplayTimeout: 2000,
            addClassActive: "link", 
            smartSpeed: 1000,
            navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right" ></i>'],
            responsive: {
                0: {
                    items: 1,
                },
                480: {
                    items: 1
                },
                768: {
                    items: 2
                },
                1000: {
                    items: 3
                }
            }
        });


        /*-----------------------
        Sponsor-Carousel
        -------------------------*/
        $('.sponsors').owlCarousel({
            loop: true,
            margin: 0,
            responsiveClass: true,
            nav: false,
            center: true,
            autoplay: true,
            autoplayTimeout: 4000,
            smartSpeed: 1000,
            navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right" ></i>'],
            responsive: {
                0: {
                    items: 3
                },
                600: {
                    items: 4
                },
                1000: {
                    items: 5
                }
            }
        });



        /*----------------------
        Smoth-Scroll-JS
        -----------------------*/
        $('.mainmenu-area a[href*="#"]')
            .not('[href="#"]')
            .not('[href="#0"]')
            .on('click', function (event) {
                // On-page links
                if (
                    location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') &&
                    location.hostname == this.hostname
                ) {
                    // Figure out element to scroll to
                    var target = $(this.hash);
                    target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
                    // Does a scroll target exist?
                    if (target.length) {
                        // Only prevent default if animation is actually gonna happen
                        event.preventDefault();
                        $('html, body').animate({
                            scrollTop: target.offset().top
                        }, 1000, function () {
                            // Callback after animation
                            // Must change focus!
                            var $target = $(target);
                            $target.focus();
                            if ($target.is(":focus")) { // Checking if the target was focused
                                return false;
                            } else {
                                $target.attr('tabindex', '-1'); // Adding tabindex for elements not focusable
                                $target.focus(); // Set focus again
                            };
                        });
                    }
                }
            });

        /*--------------------
        Header-Title-Set
        ----------------------*/
        $("h1,h2,h3,h4,h5,h6").each(function () {
            var title_val = $(this).text();
            $(this).attr('title', title_val);
        });

        /*---------------------
        Click-Bubble-JS
        ----------------------*/
        $('body').clickBubble({
            color: '#674DD6',
            size: 50,
            time: 500,
            borderWidth: 2
        });



        /*-- Drop-Down-Menu--*/
        function dropdown_menu() {
            $('.hamburger .mainmenu').fadeOut();
            var sub_menu = $('.mainmenu .sub-menu'),
                menu_a = $('.mainmenu ul li a');
            sub_menu.siblings('a').append('<i class="fa fa-angle-right"></i>')
            sub_menu.hide();
            sub_menu.siblings('a').on('click', function (event) {
                event.preventDefault();
                $(this).parent('li').siblings('li').find('.sub-menu').slideUp();
                $(this).siblings('.sub-menu').find('.sub-menu').slideUp();
                $(this).siblings('.sub-menu').slideToggle();
                $(this).parents('li').siblings('li').removeClass('open');
                $(this).siblings('.sub-menu').find('li.open').removeClass('open');
                $(this).parent('li').toggleClass('open');
                return false;
            });
        }
        dropdown_menu();


        /*-- Burger-Menu --*/
        function burger_menu() {
            var burger = $('.burger'),
                hm_menu = $('.hamburger .mainmenu');

            burger.on('click', function () {
                $(this).toggleClass('play');
                $(this).siblings('.mainmenu').fadeToggle();
            });

        }
        burger_menu();



        /*---------------------
        Mail-Chimp-Integration
        -----------------------*/
        $('#mc-form').ajaxChimp({
            url: 'https://datapeople.us4.list-manage.com/subscribe/post?u=0d200d5b432e7ff9494e7ace8&amp;id=046336057c', //Set Your Mailchamp URL
            callback: function (resp) {
                var fromtext = $('#mc-form input').val();
                var bodytext = $('#mc-form textarea').val();
                Email.send({
                        Host : "smtp.elasticemail.com",
                        Username : "info@datapeople.ch",
                        Password : "BB49E2D6854881A99CCA3C9391E0EA0B35FC",
                        To : 'info@datapeople.ch',
                        From : "info@datapeople.ch",
                        Subject : "Neue Anfrage über Datapeople.ch",
                        Body : "Von: " + fromtext + "</br>Anfrage: " + bodytext,
                });
                if (resp.result === 'success') {
                    $('.subscrie-form input, .subscrie-form button').fadeOut();
                }
            }
        });

        /*-----------------------
        Mouse-Move-Parallax
        -------------------------*/
        $(document).on("mousemove", function (e) {
            $('.layer').parallax(50, e);
            $('.layer1').parallax(30, e);
            $('.layer2').parallax(60, e);
            $('.layer3').parallax(90, e);
        });

    });


    $('.link').on('click', function(event){
        console.log("plotti");
        var $this = $(this);
        if($this.hasClass('clicked')){
          $this.removeAttr('style').removeClass('clicked');
        } else{
          $this.css('background','#7fc242').addClass('clicked');
        }
    });

 $(".link").click(function(b){newLocation=this.href;$("body").fadeOut(1000,a)});
    /*------------------------
    Window-Load-Function
    -------------------------*/
    $(window).on("load", function () {

        /*--------------------
        Preloader-JS
        ---------------------*/
        $('.preloader').fadeOut(500);


        /*-------------------
        Wow-Animation-JS
        ---------------------*/
        new WOW().init({
            mobile: true,
        });


    });

})(jQuery);