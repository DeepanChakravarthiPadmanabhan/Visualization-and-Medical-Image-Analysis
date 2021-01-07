COLOR_MAPPING = {
    'background': (0, 0, 0) ,  # black -> Background
    'C1': (255, 0, 0)    ,    # red -> Non-enhancing tumor core (NCR/NET)
    'C2': (0, 255, 0) ,  # green -> Peritumoral edema (ED)
    'C3': (0, 0, 255)   ,     # blue -> Gadolinium-enhancing tumor (ET)
    }

MODALITY = {1: 'FLAIR',
            2: 'T1',
            3: 'T1-CE',
            4: 'T2'}