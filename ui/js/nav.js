(function () {
    var navLinks = [
        { title: "Home", href: "/" },
        { title: "Module 2: Bank Marketing", href: "/module_2-bank/" },
        { title: "Tools", href: "/tools/" }
    ];

    function getBaseUrl() {
        var baseUrl = window.__SITE_BASEURL || "";
        if (baseUrl && baseUrl.endsWith("/")) {
            return baseUrl.slice(0, -1);
        }
        return baseUrl;
    }

    function buildPageUrl(relativePath) {
        return getBaseUrl() + relativePath;
    }

    function normalizePath(pathname) {
        var baseUrl = getBaseUrl();
        var path = pathname || "/";
        if (baseUrl && path.indexOf(baseUrl) === 0) {
            path = path.slice(baseUrl.length);
        }
        if (!path.endsWith("/")) {
            path += "/";
        }
        return path;
    }

    function buildNavList() {
        var nav = document.createElement("nav");
        nav.className = "corner-menu-nav";
        nav.setAttribute("aria-label", "Site navigation");

        var currentPath = normalizePath(window.location.pathname);

        navLinks.forEach(function (item) {
            var link = document.createElement("a");
            var targetPath = normalizePath(item.href);
            link.href = buildPageUrl(item.href);
            link.textContent = item.title;
            if (currentPath === targetPath) {
                link.classList.add("is-active");
            }
            nav.appendChild(link);
        });

        return nav;
    }

    function buildMenu() {
        if (document.querySelector(".corner-menu")) {
            return;
        }

        var container = document.createElement("div");
        container.className = "corner-menu";
        container.setAttribute("data-pinned", "false");

        var toggleButton = document.createElement("button");
        toggleButton.className = "corner-menu-toggle";
        toggleButton.type = "button";
        toggleButton.setAttribute("aria-label", "Open site menu");
        toggleButton.setAttribute("aria-haspopup", "true");
        toggleButton.setAttribute("aria-expanded", "false");

        var labelSpan = document.createElement("span");
        labelSpan.textContent = "Menu";
        var iconSpan = document.createElement("span");
        iconSpan.setAttribute("aria-hidden", "true");
        iconSpan.textContent = "\u2630";
        toggleButton.appendChild(labelSpan);
        toggleButton.appendChild(iconSpan);

        var panel = document.createElement("div");
        panel.className = "corner-menu-panel";

        var navSection = document.createElement("section");
        navSection.className = "corner-menu-section";

        var navTitle = document.createElement("p");
        navTitle.className = "corner-menu-title";
        navTitle.textContent = "Navigate";

        navSection.appendChild(navTitle);
        navSection.appendChild(buildNavList());
        panel.appendChild(navSection);

        function isPinned() {
            return container.getAttribute("data-pinned") === "true";
        }

        function syncExpandedState() {
            var expanded = isPinned() || container.matches(":hover") || container.matches(":focus-within");
            toggleButton.setAttribute("aria-expanded", String(expanded));
        }

        function setPinned(nextPinned) {
            container.setAttribute("data-pinned", String(nextPinned));
            syncExpandedState();
        }

        toggleButton.addEventListener("click", function () {
            setPinned(!isPinned());
        });

        document.addEventListener("click", function (event) {
            if (!container.contains(event.target)) {
                setPinned(false);
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                setPinned(false);
            }
        });

        container.addEventListener("mouseenter", syncExpandedState);
        container.addEventListener("mouseleave", syncExpandedState);
        container.addEventListener("focusin", syncExpandedState);
        container.addEventListener("focusout", function (event) {
            var nextTarget = event.relatedTarget;
            if (nextTarget && container.contains(nextTarget)) {
                return;
            }
            syncExpandedState();
        });

        panel.addEventListener("click", function (event) {
            if (event.target && event.target.closest("a")) {
                setPinned(false);
            }
        });

        container.appendChild(toggleButton);
        container.appendChild(panel);
        document.body.appendChild(container);
        syncExpandedState();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", buildMenu);
    } else {
        buildMenu();
    }
})();
