from biobeam import focus_field_beam


u = focus_field_beam(shape = (128,128,512),
                     units = (0.1,0.1,.01),
                     NA = 0.6,
                     n0 = 1.33)




