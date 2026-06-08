/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./templates/**/*.html"],
    plugins: [require("daisyui")],
    daisyui: {
        themes: [{
            paper: {
                "primary": "#1f2937",
                "primary-content": "#FEFEFE",
                "secondary": "#1f2937",
                "secondary-content": "#FEFEFE",
                "accent": "#FFCB2E",
                "accent-content": "#1f2937",
                "neutral": "#FEFEFE",
                "neutral-content": "#1f2937",
                "base-100": "#FEFEFE",
                "base-200": "#F5F5F5",
                "base-300": "#EFEFEF",
                "base-content": "#1f2937",
                "info": "#1f2937",
                "info-content": "#FEFEFE",
                "success": "#81CFD1",
                "success-content": "#FEFEFE",
                "warning": "#EFD7BB",
                "warning-content": "#1f2937",
                "error": "#E58B8B",
                "error-content": "#FEFEFE",
            },
        }],
    },
};
