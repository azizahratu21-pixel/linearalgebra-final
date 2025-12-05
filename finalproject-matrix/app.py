# app.py
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Matrix Transformations in Image Processing", layout="wide")

page_bg = """
<style>

html, body, [class*="css"]  {
    background-color: #DDE5D5 !important;   
}

/* App container background */
.stApp {
    background-color: #D3E3D2<p style="position: absolute; z-index: 10; top: 12px; left: 84px; transform: translate(0%, -50%); display: flex; justify-content: center; align-items: center; background-color: var(--col-body-bg2); padding: 2px 4px; margin: auto; border-radius: 5px; border: 1px solid var(--col-accent2); text-transform: capitalize; font-size: 12px; height: 24px;">copied</p> !important;
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

st.title("Matrix Transformations in Image Processing")
st.markdown(
    """
This multi-page app demonstrates **geometric transformations** implemented with 3×3 homogeneous matrices
(translation, scaling, rotation, shearing, reflection) and **convolution-based filters** (blur, sharpen)
implemented manually with kernels.

**How to use**
- Go to **Image Processing** page to upload an image, choose transformations/filters and preview results.
- The transformations are applied using matrices (implemented manually) with inverse mapping + bilinear sampling.
- Blur & Sharpen use manual convolution kernels (no built-in blur functions).

**Learning goals**
- Understand how 2D affine transformations are represented with matrices (homogeneous coordinates).
- Practice writing convolution kernels and applying them to images.
    """
)

st.header("Matrix transformations (brief)")
st.markdown(
    """
- Use 3×3 matrices on homogeneous coordinates `[x, y, 1]` to combine translation with linear transforms.
- Order matters — matrix multiplication is not commutative.
- For image warping we use inverse mapping: for each destination pixel, compute source coordinates via inverse transform and sample the image (bilinear interpolation).
"""
)

st.header("Convolution (brief)")
st.markdown(
    """
- Convolution is implemented by sliding a kernel over the image and summing products.
- Example kernels:
    - Blur (3×3 average): `1/9 * ones((3,3))`
    - Sharpen: `[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]`
"""
)

st.write("---")
st.info("Now go to the **Image Processing** page to try transforms and filters.")
