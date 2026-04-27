
unique_intra_label_mapping = {
                                # General (Vanilla Case)
                                "general" : 0,


                                # Dissolve Effect
                                "dissolve" : 1,
                                "cross_blur" : 1,
                                "ripple_dissolve" : 1,


                                # Wipe Effect       
                                "wipe" : 2,
                                "wipe_diagonal" : 2,
                                "wipe_spin" : 2,
                                "wipe_circle_open" : 2,
                                "wipe_circle_close" : 2,
                                "wipe_bars" : 2,
                                "ripple_open" : 2,
                                "page_curl_left" : 2,
                                "page_curl_right" : 2,
                                "mosaic" : 2,


                                # Push Effect      
                                "push" : 3,
                                "puzzle_blend" : 3,


                                # Slide Effect      
                                "slide" : 4,
                                "whip_pan" : 4,
                                "cube" : 4,                     


                                # Zoom Effect      
                                "zoom_in" : 5,
                                "zoom_out" : 5,
                                "spin_in" : 5,
                                "spin_out" : 5,
                                "swap" : 5,
                                "cross_zoom" : 5,


                                # Fade Effect in the middle transition 
                                "fadeAtoblack" : 6,             
                                "fadeAtowhite" : 6,             
                                "fadeBfromblack" : 6,           
                                "fadeBfromwhite" : 6,           
                                "dip_to_black" : 6,             
                                "dip_to_white" : 6,             
                                "fade_in" : 6,                  
                                "fade_out" : 6,                 


                                # Doorway Effect   
                                "doorway" : 7, 
                                "srcA_away": 7,                 


                                # Padding
                                "padding" : 8,                 


                            }

# For the Benchmark convert
inverse_intra_label_mapping = {
                                    "General" : 0,
                                    "Dissolve Effect" : 1,
                                    "Wipe Effect" : 2,
                                    "Fancy Wipes" : 2,      # duplicate (our label bug)
                                    "Push Effect" : 3,
                                    "Slide Effect" : 4, 
                                    "Zoom Effect" : 5,
                                    "Fade Effect" : 6,
                                    "Doorway Effect" : 7,
                                }





#############################################  Inter Label  ############################################

unique_inter_label_mapping = {

                                "new_start" : 0,           
                                "hard_cut" : 1,             
                                "transition_source" : 2,    
                                "transition" : 3,           
                                "sudden_jump" : 4,         

                                # Padding
                                "padding" : 5,            
                    
                            }

inverse_inter_label_mapping = {
                                "New_Start" : 0,
                                "Hard_Cut" : 1,
                                "Transition" : 3,
                                "Sudden_Jump" : 4,
                            }