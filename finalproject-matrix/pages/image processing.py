import streamlit as st
from PIL import Image
import numpy as np
import io
import math

st.set_page_config(layout="wide")
st.title("Image Processing Tools")

page_bg = """
<style>

html, body, [class*="css"]  {
    background-color: #DDE5D5 !important;   
}

/* App container background */
.stApp {
    background-color: #DDE5D5 !important;
}

/* Remove white header */
header[data-testid="stHeader"] {
    background-color: #DDE5D5 !important;
}

/* Main content */
main[data-testid="stMain"] {
    background-color: #DDE5D5 !important;
    padding-top: 0 !important;
}

/* Main block container */
.block-container {
    background-color: #DDE5D5 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #DDE5D5 !important;
}

/* Darker sage section box */
.feature-box {
    background-color: #C3D1C0;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------
# Utility functions
# -----------------------
def pil_to_np(img: Image.Image):
    arr = np.array(img.convert("RGBA")).astype(np.float32) / 255.0
    return arr  # H x W x 4 (RGBA)

def np_to_pil(arr: np.ndarray):
    arr8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGBA")

def make_translation(tx, ty):
    M = np.eye(3)
    M[0,2] = tx
    M[1,2] = ty
    return M

def make_scaling(sx, sy):
    M = np.eye(3)
    M[0,0] = sx
    M[1,1] = sy
    return M

def make_rotation(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    M = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=float)
    return M

def make_shear(shx, shy):
    M = np.eye(3)
    M[0,1] = shx  # x' = x + shx*y
    M[1,0] = shy  # y' = y + shy*x
    return M

def make_reflection(axis):
    if axis == "x-axis":
        return np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=float)
    if axis == "y-axis":
        return np.array([[-1,0,0],[0,1,0],[0,0,1]], dtype=float)
    if axis == "origin":
        return np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=float)
    if axis == "y=x":
        return np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=float)
    return np.eye(3)

# Bilinear sampling
def bilinear_sample(img, x, y):
    h, w = img.shape[:2]
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        # outside -> return transparent
        return np.array([0,0,0,0], dtype=np.float32)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0
    top = (1-dx) * img[y0, x0] + dx * img[y0, x1]
    bottom = (1-dx) * img[y1, x0] + dx * img[y1, x1]
    return (1-dy) * top + dy * bottom

# Inverse warp using 3x3 matrix M (destination coords -> source coords via inv(M))
def warp_image(src_img, M, out_shape=None, fill_color=(0,0,0,0)):
    h, w = src_img.shape[:2]
    if out_shape is None:
        out_h, out_w = h, w
    else:
        out_h, out_w = out_shape
    dst = np.zeros((out_h, out_w, 4), dtype=np.float32)
    invM = np.linalg.inv(M)
    # Loop over destination pixels
    for j in range(out_h):
        for i in range(out_w):
            # destination coordinate (i,j) -> map to source coordinate
            dst_hom = np.array([i, j, 1.0], dtype=float)
            src_hom = invM @ dst_hom
            x_src, y_src = src_hom[0], src_hom[1]
            px = bilinear_sample(src_img, x_src, y_src)
            dst[j, i] = px
    return dst

# Convolution (per-channel)
def conv2d_image(img, kernel):
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    h, w = img.shape[:2]
    # pad with edge values for each channel (exclude alpha separately)
    rgb = img[..., :3]
    alpha = img[..., 3:]
    padded = np.pad(rgb, ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='edge')
    out = np.zeros_like(rgb)
    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw, :]  # kh x kw x 3
            # element-wise multiply and sum across kernel
            res = (region * kernel[..., None]).sum(axis=(0,1))
            out[y,x] = res
    # clamp and attach alpha unchanged
    out = np.clip(out, 0.0, 1.0)
    return np.concatenate([out, alpha], axis=2)

# -----------------------
# UI: Upload + Controls
# -----------------------
col1, col2 = st.columns([1,2])
with col1:
    uploaded = st.file_uploader("Upload image (PNG/JPG)", type=['png','jpg','jpeg'])
    st.markdown("**Transform order**: choose the order transforms are applied (left-to-right).")
    order_mode = st.radio("Order mode:", ("Preset: Translation→Scale→Rotate→Shear→Reflect", "Custom order"))
    if order_mode.startswith("Custom"):
        choices = ["Translation","Scaling","Rotation","Shearing","Reflection"]
        order = st.multiselect("Select transforms in desired order", choices, default=choices)
    else:
        order = ["Translation","Scaling","Rotation","Shearing","Reflection"]

    st.markdown("---")
    st.subheader("Translation")
    t_enable = st.checkbox("Enable translation", value=False)
    tx = st.number_input("tx (pixels)", value=0.0, format="%.1f")
    ty = st.number_input("ty (pixels)", value=0.0, format="%.1f")

    st.subheader("Scaling")
    s_enable = st.checkbox("Enable scaling", value=False)
    sx = st.number_input("sx", value=1.0, format="%.3f")
    sy = st.number_input("sy", value=1.0, format="%.3f")
    scale_about_center = st.checkbox("Scale about image center", value=True)

    st.subheader("Rotation")
    r_enable = st.checkbox("Enable rotation", value=False)
    angle = st.slider("Angle (degrees)", -360, 360, 0)

    st.subheader("Shearing")
    sh_enable = st.checkbox("Enable shearing", value=False)
    shx = st.number_input("shear x (shx)", value=0.0, format="%.3f")
    shy = st.number_input("shear y (shy)", value=0.0, format="%.3f")

    st.subheader("Reflection")
    rf_enable = st.checkbox("Enable reflection", value=False)
    rf_axis = st.selectbox("Axis/line", ["x-axis","y-axis","origin","y=x"])

    st.markdown("---")
    st.subheader("Filters (Convolution)")
    apply_blur = st.checkbox("Blur (3×3 average)", value=False)
    apply_sharpen = st.checkbox("Sharpen", value=False)

    st.markdown("---")
    st.subheader("Simple background removal (HSV threshold)")
    do_bgrem = st.checkbox("Enable simple background removal (color threshold)", value=False)
    if do_bgrem:
        st.markdown("Pick a sample color and tolerance to mask background.")
        # Provide default for green-screen like removal
        hue_low = st.slider("Hue low", 0, 180, 35)
        hue_high = st.slider("Hue high", 0, 180, 85)
        sat_low = st.slider("Sat low", 0, 255, 40)
        val_low = st.slider("Val low", 0, 255, 40)

    st.markdown("---")
    if st.button("Apply operations"):
        do_apply = True
    else:
        do_apply = False

with col2:
    st.subheader("Preview")
    if uploaded is None:
        st.info("Upload an image to start. Recommended: small image (e.g., 400×400) for faster processing.")
    else:
        image = Image.open(uploaded).convert("RGBA")
        src = pil_to_np(image)  # float32 normalized
        h, w = src.shape[:2]

        # Background removal (simple HSV threshold) - optional
        if do_bgrem:
            # convert to HSV via colorsys-ish (approx using OpenCV formula, but avoid dependency)
            rgb = (src[..., :3] * 255).astype(np.uint8)
            # simple conversion using PIL
            pil_rgb = Image.fromarray(rgb, mode='RGB')
            hsv = np.array(pil_rgb.convert('HSV')).astype(np.uint8)  # H:0-255 scale (PIL)
            # convert H scale 0-255 to 0-180 approximate (OpenCV uses 0-180)
            H = (hsv[...,0].astype(int) * 180) // 255
            S = hsv[...,1]
            V = hsv[...,2]
            mask = ((H >= hue_low) & (H <= hue_high) & (S >= sat_low) & (V >= val_low))
            # set masked pixels alpha = 0
            src[..., 3][mask] = 0.0

        # Build total transform matrix (start with identity)
        M_total = np.eye(3)
        # Helper: apply about center
        cx, cy = (w-1)/2.0, (h-1)/2.0

        for step in order:
            if step == "Translation" and t_enable:
                M = make_translation(tx, ty)
                M_total = M @ M_total
            if step == "Scaling" and s_enable:
                if scale_about_center:
                    T1 = make_translation(-cx, -cy)
                    S = make_scaling(sx, sy)
                    T2 = make_translation(cx, cy)
                    M = T2 @ S @ T1
                else:
                    M = make_scaling(sx, sy)
                M_total = M @ M_total
            if step == "Rotation" and r_enable:
                # rotate about center
                T1 = make_translation(-cx, -cy)
                R = make_rotation(angle)
                T2 = make_translation(cx, cy)
                M = T2 @ R @ T1
                M_total = M @ M_total
            if step == "Shearing" and sh_enable:
                M = make_shear(shx, shy)
                M_total = M @ M_total
            if step == "Reflection" and rf_enable:
                M = make_reflection(rf_axis)
                # reflect about center if axis is not origin-based? we will reflect about image center for better UX
                T1 = make_translation(-cx, -cy)
                T2 = make_translation(cx, cy)
                M_total = (T2 @ M @ T1) @ M_total

        # If nothing to apply and no filters, just show original
        if not do_apply and not apply_blur and not apply_sharpen:
            st.image(np_to_pil(src), caption="Original")
        else:
            # Warp using M_total
            # Determine output canvas size: keep same for simplicity
            out = warp_image(src, M_total, out_shape=(h, w))
            # Apply convolution filters
            if apply_blur:
                kernel_blur = np.ones((3,3), dtype=np.float32) / 9.0
                out = conv2d_image(out, kernel_blur)
            if apply_sharpen:
                kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
                out = conv2d_image(out, kernel_sharp)

            # Show side-by-side original vs transformed
            colA, colB = st.columns(2)
            with colA:
                st.image(np_to_pil(src), caption="Original (RGBA)")
            with colB:
                st.image(np_to_pil(out), caption="Transformed / Filtered (RGBA)")

            st.markdown("**Transformation matrix (3×3) applied:**")
            st.write(np.round(M_total, 6))
            st.markdown("**Notes:**")
            st.write("- Warping uses inverse mapping + bilinear sampling.")
            st.write("- Convolution implemented manually on RGB channels (alpha preserved).")