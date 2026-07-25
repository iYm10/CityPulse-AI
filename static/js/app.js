document.addEventListener("DOMContentLoaded", () => {
    const menuButton = document.querySelector(".menu-toggle");
    const nav = document.querySelector(".nav-links");

    if (menuButton && nav) {
        menuButton.addEventListener("click", () => {
            nav.classList.toggle("open");
        });
    }

    document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab").forEach((item) => {
                item.classList.remove("active");
            });

            document.querySelectorAll(".tab-panel").forEach((panel) => {
                panel.classList.remove("active");
            });

            button.classList.add("active");
            document.getElementById(button.dataset.tab)?.classList.add("active");
        });
    });

    const chartCanvas = document.getElementById("wasteChart");

    if (window.cityPulseWasteData && chartCanvas) {
        new Chart(chartCanvas, {
            type: "bar",
            data: {
                labels: window.cityPulseWasteData.labels,
                datasets: [
                    {
                        data: window.cityPulseWasteData.values,
                        borderRadius: 12,
                        backgroundColor: ["#c7d3e5", "#7893b8", "#0b7b7c"],
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: "rgba(17,40,70,.08)",
                        },
                    },
                },
            },
        });
    }
});
