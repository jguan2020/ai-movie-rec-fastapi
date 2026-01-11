const form = document.getElementById("search-form");
const loader = document.getElementById("loader");

if (form && loader) {
  form.addEventListener("submit", () => {
    loader.classList.remove("hidden");
  });
}

const favoriteButtons = document.querySelectorAll(".favorite-toggle");

favoriteButtons.forEach((button) => {
  button.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (button.dataset.auth !== "true") {
      window.location.href = "/login";
      return;
    }

    if (button.dataset.enabled !== "true") {
      return;
    }

    if (button.disabled) {
      return;
    }

    button.disabled = true;

    const payload = {
      title: button.dataset.title || "",
      release_date: button.dataset.releaseDate || "",
      poster_path: button.dataset.posterPath || "",
      genres: button.dataset.genres || "",
    };

    try {
      const response = await fetch("/favorite", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (response.status === 401) {
        window.location.href = "/login";
        return;
      }

      const data = await response.json();
      if (response.status === 403 && data.error === "favorites_limit") {
        alert("Free tier limit is 10 favorites. Upgrade to Premium to add more.");
        return;
      }
      if (response.ok && typeof data.favorited === "boolean") {
        button.classList.toggle("active", data.favorited);
      }
    } catch (error) {
      console.error("Favorite toggle failed", error);
    } finally {
      button.disabled = false;
    }
  });
});

const modal = document.getElementById("movie-modal");
const modalPoster = document.getElementById("modal-poster");
const modalPlaceholder = document.getElementById("modal-placeholder");
const modalTitle = document.getElementById("modal-title");
const modalRelease = document.getElementById("modal-release");
const modalRating = document.getElementById("modal-rating");
const modalRuntime = document.getElementById("modal-runtime");
const modalGenres = document.getElementById("modal-genres");
const modalKeywords = document.getElementById("modal-keywords");
const modalMatches = document.getElementById("modal-matches");
const modalOverview = document.getElementById("modal-overview");
const modalOverviewLocked = document.getElementById("modal-overview-locked");
const modalCloseButtons = document.querySelectorAll("[data-modal-close]");

const setModalField = (element, value) => {
  if (!element) {
    return;
  }
  const item = element.closest(".modal-item");
  if (value) {
    element.textContent = value;
    if (item) {
      item.classList.remove("hidden");
    }
  } else {
    element.textContent = "";
    if (item) {
      item.classList.add("hidden");
    }
  }
};

const openModal = (data) => {
  if (!modal) {
    return;
  }
  if (modalTitle) {
    modalTitle.textContent = data.title || "";
  }

  if (modalPoster && modalPlaceholder) {
    if (data.posterUrl) {
      modalPoster.src = data.posterUrl;
      modalPoster.alt = data.title || "Movie poster";
      modalPoster.style.display = "block";
      modalPlaceholder.style.display = "none";
    } else {
      modalPoster.removeAttribute("src");
      modalPoster.alt = "";
      modalPoster.style.display = "none";
      modalPlaceholder.style.display = "flex";
    }
  }

  setModalField(modalRelease, data.releaseDate);
  setModalField(modalRating, data.rating);
  setModalField(modalRuntime, data.runtime ? `${data.runtime} min` : "");
  setModalField(modalGenres, data.genres);
  setModalField(modalKeywords, data.keywords);
  setModalField(modalMatches, data.matched);

  if (data.canOverview) {
    if (modalOverviewLocked) {
      modalOverviewLocked.classList.add("hidden");
    }
    if (modalOverview) {
      modalOverview.textContent = data.overview || "No overview available.";
      modalOverview.classList.remove("hidden");
    }
  } else {
    if (modalOverview) {
      modalOverview.textContent = "";
      modalOverview.classList.add("hidden");
    }
    if (modalOverviewLocked) {
      modalOverviewLocked.classList.remove("hidden");
    }
  }

  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
};

const closeModal = () => {
  if (!modal) {
    return;
  }
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
};

modalCloseButtons.forEach((button) => {
  button.addEventListener("click", () => closeModal());
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
  }
});

const movieRows = document.querySelectorAll(".movie-row");
const overviewLinks = document.querySelectorAll(".overview-link");

overviewLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.stopPropagation();
  });
});

movieRows.forEach((row) => {
  row.addEventListener("click", () => {
    openModal({
      title: row.dataset.title || "",
      releaseDate: row.dataset.releaseDate || "",
      rating: row.dataset.rating || "",
      runtime: row.dataset.runtime || "",
      genres: row.dataset.genres || "",
      keywords: row.dataset.keywords || "",
      matched: row.dataset.matched || "",
      overview: row.dataset.overview || "",
      canOverview: row.dataset.canOverview === "true",
      posterUrl: row.dataset.posterUrl || "",
    });
  });

  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      row.click();
    }
  });
});
