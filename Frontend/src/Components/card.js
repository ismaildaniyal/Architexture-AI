import React, { useState, useEffect } from "react";

const ProCard = () => {
  const [cardStyles, setCardStyles] = useState(getStyles());

  // Function to get styles based on window width
  function getStyles() {
    const width = typeof window !== 'undefined' ? window.innerWidth : 1024;
    let styles = {
      cardContainer: {
        background: "linear-gradient(135deg, #6a6af0, #8e72f1)",
        borderRadius: "15px",
        padding: "12px",
        textAlign: "center",
        width: "180px",
        color: "#fff",
        boxShadow: "0px 10px 20px rgba(0, 0, 0, 0.1)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        minHeight: "160px",
      },
      contentWrapper: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        flex: 1,
      },
      iconContainer: {
        backgroundColor: "#ffffff",
        borderRadius: "50%",
        width: "40px",
        height: "40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "8px",
      },
      icon: {
        width: "20px",
        height: "20px",
        backgroundColor: "transparent",
        borderRadius: "50%",
        border: "3px solid #6a6af0",
      },
      heading: {
        fontSize: "15px",
        fontWeight: "bold",
        margin: "6px 0",
      },
      text: {
        fontSize: "12px",
        margin: "6px 0",
        lineHeight: "1.3",
      },
      button: {
        background: "#ffffff",
        color: "#6a6af0",
        border: "none",
        borderRadius: "8px",
        padding: "6px 12px",
        cursor: "pointer",
        fontWeight: "bold",
        fontSize: "11px",
        transition: "all 0.3s ease",
        marginTop: "8px",
        width: "80%",
        alignSelf: "center",
      },
    };

    if (width <= 480) {
      styles.cardContainer.width = "160px";
      styles.cardContainer.minHeight = "140px";
      styles.iconContainer.width = "35px";
      styles.iconContainer.height = "35px";
      styles.icon.width = "18px";
      styles.icon.height = "18px";
      styles.heading.fontSize = "13px";
      styles.text.fontSize = "11px";
      styles.button.fontSize = "10px";
      styles.button.padding = "5px 10px";
    }
    else if (width <= 768) {
      styles.cardContainer.width = "170px";
      styles.cardContainer.minHeight = "150px";
      styles.heading.fontSize = "14px";
      styles.text.fontSize = "11px";
      styles.button.fontSize = "10px";
    }
    else if (width >= 1440) {
      styles.cardContainer.width = "200px";
      styles.cardContainer.minHeight = "180px";
      styles.iconContainer.width = "45px";
      styles.iconContainer.height = "45px";
      styles.heading.fontSize = "16px";
      styles.text.fontSize = "13px";
      styles.button.fontSize = "12px";
      styles.button.padding = "8px 16px";
    }

    return styles;
  }

  // Add resize event listener
  useEffect(() => {
    const handleResize = () => {
      setCardStyles(getStyles());
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div style={cardStyles.cardContainer}>
      <div style={cardStyles.contentWrapper}>
        <div style={cardStyles.iconContainer}>
          <div style={cardStyles.icon}></div>
        </div>
        <h3 style={cardStyles.heading}>Upgrade to PRO</h3>
        <p style={cardStyles.text}>
          Get unlimited design genrations
        </p>
      </div>
      <button style={cardStyles.button}>Get started with PRO</button>
    </div>
  );
};

export default ProCard;
