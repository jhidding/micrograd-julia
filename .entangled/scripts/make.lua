local system = require 'pandoc.system'

local make_preamble = [[
.RECIPEPREFIX := >

target_dir := %s
target := $(target_dir)/%s
]]

function CodeBlock(block)
    if block.classes[1] == "make" then
        if block.classes[2] == "figure" then
            local caption, rawsrc = block.text:match("(.-)\n%-%-%-\n(.*)")
            if not caption then
                rawsrc = block.text
                caption = ""
            end
            outfile = block.attributes["target"]
            system.with_temporary_directory("run-make", function (tmpdir)
                local src = make_preamble:format("docs", outfile) .. rawsrc
                local f = io.open(tmpdir .. "/Makefile", "w")
                f:write(src)
                f:close()
                os.execute("mkdir -p docs/$(dirname " .. outfile ..")")
                os.execute("make -f " .. tmpdir .. "/Makefile " .. "docs/" .. outfile)
            end)
            return pandoc.Para({pandoc.Image({pandoc.Str(caption)}, outfile, caption, {class = "figure"})})
        end

        if block.classes[2] == "include" then
            local caption, rawsrc = block.text:match("(.-)\n%-%-%-\n(.*)")
            if not caption then
                rawsrc = block.text
                caption = ""
            end
            outfile = block.attributes["target"]
            os.execute("mkdir -p $(dirname " .. outfile .. ")")
            system.with_temporary_directory("run-make", function (tmpdir)
                local src = make_preamble:format(".", outfile) .. rawsrc
                local f = io.open(tmpdir .. "/Makefile", "w")
                f:write(src)
                f:close()
                os.execute("make -f " .. tmpdir .. "/Makefile " .. outfile)
            end)
            local f = io.open(outfile, "r")
            local content = f:read("*a")
            return pandoc.CodeBlock(content, {class = "output"})
        end
    end
end

