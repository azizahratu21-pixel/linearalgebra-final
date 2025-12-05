# pages/2_Team.py
import streamlit as st
from PIL import Image
import os

st.set_page_config(layout="centered")
st.title("Team Members")

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

st.markdown("Here are the team members along with their roles and contributions in completing this project.")

members = [
    {
        "name": "Inayah Ratu Azizah",
        "role": "Project Manager",
        "photo": "assets/member1.jpeg",
        "desc": (
            "Managed the project timeline, assigned tasks to team members, and designed the interface "
            "layout in the Streamlit app for better usability. Responsible for multi-page navigation, "
            "sidebar layout, and visual comfort of the application."
        )
    },
    {
        "name": "Ahmad Raihanun Nabil",
        "role": "Matrix Transformation Developer",
        "photo": "assets/member2.jpeg",
        "desc": (
            "Implemented all matrix-based transformations such as translation, scaling, rotation, shearing, "
            "and reflection using 3×3 homogeneous matrices. Also developed inverse mapping and bilinear "
            "interpolation functions for the image warping process."
        )
    },
    {
        "name": "Novita Aulia",
        "role": "Image Processing Engineer (Convolution)",
        "photo": "assets/member3.jpeg",
        "desc": (
            "Developed filtering features using manual convolution operations, including blur "
            "(smoothing filter) and sharpen (high-pass filter). Designed 3×3 kernels and ensured "
            "filters work across all RGB channels."
        )
    },
    {
        "name": "Windi Melisa Sipayung",
        "role": "Deployment & Report Specialist",
        "photo": "assets/member4.jpeg",
        "desc": (
            "Managed application deployment to Streamlit Cloud, organized the GitHub repository, created "
            "requirements.txt, and conducted final testing before the demo. Prepared PDF reports containing "
            "feature explanations, documentation, and application screenshots."
        )
    }
]

# Display members
for m in members:
    cols = st.columns([1, 3])
    with cols[0]:
        if os.path.exists(m["photo"]):
            st.image(m["photo"], width=130)
        else:
            st.image(Image.new("RGBA", (130, 130), (200, 200, 200, 255)), caption="No photo")
    with cols[1]:
        st.subheader(m["name"])
        st.write(f"**Role:** {m['role']}")
        st.write(m["desc"])
    st.write("---")

st.markdown("### Application Workflow (Summary)")
st.write(
    """
This application consists of three main pages:

1. **Home Page**  
   Provides a basic explanation of matrix transformations and convolution, along with a brief visualization.

2. **Image Processing Tools**  
   Users can upload images, select transformations (translation, scaling, rotation, shearing, reflection), 
   and apply convolution filters (blur & sharpen). Transformations are applied using 3×3 homogeneous matrices 
   and image warping is done with inverse mapping. Filtering uses manual kernels.

3. **Team Members**  
   Displays the team members along with their roles and contributions.

All features are implemented using Python, NumPy, Streamlit for matrix operations 
and convolution. CSS is used for the visualization coloring.
"""
)