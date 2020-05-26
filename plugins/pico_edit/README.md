Pico Edit plugin
================
Plugin provides a small admin panel for editing page content and CMS configuration.

Install
-------

1. Set permissions of `config/` and `content/` directories to allow read/write (777 or 755 recommended).
2. Clone the Github repo into your 'plugins' directory (so you get a 'pico_edit' subdirectory) OR download source code from releases and put it into 'pico_edit' subdir on 'plugins' directory.
3. Open your config.yml file on configuration and add following lines to it:
```
	pico_edit_password: put here SHA256 hashed password
	pico_edit_default_author: put here default author to add for YAML header when creating page
```
4. Visit https://www.yoursite.com/pico_edit and login

**Pay attention** if you are installing plugin on a fresh CMS, make sure that you have *config.yml* file and not *config.yml.template*. For well working Pico Edit requires existing configuration file.

About
-----
Pico Edit's features:

* Simple and clean interface
* Page create/edit/delete
* Markdown preview (top right icon in the editor)
* Edit 404 page (aka "page not found")
* Edit Pico's config.yml file.
* Some basic Git functions such as push, pull etc.

![Screenshot](https://github.com/Nepose/pico_edit/blob/master/screenshot.png)

Deny access to some files
-------------------------
If you want to deny editing some file (example configuration file or index page), just set its permissions to read only. "Saved" will show, but the content will not change.

Git functions
-------------

The general use-case is for one or more content editors to have a Git repo cloned onto their laptops. They can then go ahead and create or edit content, saving it to their local machine as required. When they're happy, they can commit to their local Git repo. How they publish the content is up to the administrator, but one method is to have a post-update hook on the Git server that publishes the content into the DocumentRoot of the webserver(s). Obviously, editors can Git-pull the changes other editors have made to their local machines so that they stay up to date. Depending on how you set things up, it's possible editors could even edit pages directly on the public website (and commit those to the Git repo from there).

Git features are only shown in the editor UI if the server has a Git binary available, and the content is in a Git repo. Push/pull functions are only available if the repo has one or more remote servers configured into it.

History
-------

* This Pico Edit is a fork + modification of [Pico Edit done by blocknotes](https://github.com/blocknotes/pico_edit). It contains possibility to direct edit config.yml instead options.conf file for overwriting.

* Pico Edit is a fork + modifications of [Peeked](https://github.com/coofercat/peeked). It contains minor improvements and some new feature like ability to edit 404 page and Pico options.

* Peeked is a fork + modifications of the [Pico Editor](https://github.com/gilbitron/Pico-Editor-Plugin), written by [Gilbert Pellegrom](https://github.com/gilbitron). It contains a few bug fixes and some functional changes, most particularly the addition of some Git capabilities.
