const upgradeButton = document.getElementById("upgrade-button");
const checkoutOverlay = document.getElementById("checkout-overlay");
const checkoutMount = document.getElementById("checkout-mount");
const checkoutLoading = document.getElementById("checkout-loading");
const checkoutError = document.getElementById("checkout-error");
const checkoutCloseButtons = document.querySelectorAll("[data-checkout-close]");

let stripeClient = null;
let embeddedCheckout = null;

const showCheckoutError = (message) => {
  if (!checkoutError) {
    return;
  }
  checkoutError.textContent = message;
  checkoutError.classList.remove("hidden");
};

const clearCheckoutError = () => {
  if (!checkoutError) {
    return;
  }
  checkoutError.textContent = "";
  checkoutError.classList.add("hidden");
};

const resetCheckout = () => {
  if (embeddedCheckout) {
    embeddedCheckout.destroy();
    embeddedCheckout = null;
  }
  if (checkoutMount) {
    checkoutMount.innerHTML = "";
  }
  if (checkoutLoading) {
    checkoutLoading.classList.add("hidden");
  }
};

const openCheckout = async () => {
  if (!checkoutOverlay) {
    return;
  }
  clearCheckoutError();
  checkoutOverlay.classList.add("open");
  checkoutOverlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");

  if (checkoutLoading) {
    checkoutLoading.classList.remove("hidden");
  }

  const publishableKey = document.body.dataset.stripeKey;
  if (!publishableKey) {
    showCheckoutError("Stripe is not configured yet.");
    if (checkoutLoading) {
      checkoutLoading.classList.add("hidden");
    }
    return;
  }

  if (!window.Stripe) {
    showCheckoutError("Stripe.js failed to load.");
    if (checkoutLoading) {
      checkoutLoading.classList.add("hidden");
    }
    return;
  }

  if (!stripeClient) {
    stripeClient = Stripe(publishableKey);
  }

  try {
    const response = await fetch("/subscribe", { method: "POST" });
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }
    const data = await response.json();
    if (!response.ok || !data.clientSecret) {
      throw new Error(data.error || "stripe_session");
    }

    embeddedCheckout = await stripeClient.initEmbeddedCheckout({
      clientSecret: data.clientSecret,
    });
    if (checkoutLoading) {
      checkoutLoading.classList.add("hidden");
    }
    embeddedCheckout.mount("#checkout-mount");
  } catch (error) {
    if (checkoutLoading) {
      checkoutLoading.classList.add("hidden");
    }
    showCheckoutError("Could not start checkout. Please try again.");
  }
};

const closeCheckout = () => {
  if (!checkoutOverlay) {
    return;
  }
  checkoutOverlay.classList.remove("open");
  checkoutOverlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
  resetCheckout();
};

if (upgradeButton) {
  upgradeButton.addEventListener("click", openCheckout);
}

checkoutCloseButtons.forEach((button) => {
  button.addEventListener("click", closeCheckout);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && checkoutOverlay?.classList.contains("open")) {
    closeCheckout();
  }
});
