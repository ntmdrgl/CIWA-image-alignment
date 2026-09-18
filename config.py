from types import SimpleNamespace

CFG = SimpleNamespace(
    # --- Image Geometric Transformations ---
    
    # Affine Transformation
    angle=0,    # angle of rotation
    hx=0,       # shear x
    hy=0,       # shear y
    sx=1,       # scale x
    sy=1,       # scale y
    tx=0,       # translation x
    ty=0,       # translation y

    # Image Distortion
    
    source_path="checkerboard.png"
)